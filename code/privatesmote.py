"""Apply interpolation with PrivateSMOTE
This script will apply PrivateSMOTE technique in the k highest-risk cases.
"""
# Import necessary libraries
from os import sep
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse


def keep_numbers(df):
    """Fix data types according to the data"""
    for col in df.columns:
        # Transform numerical strings to digits
        if isinstance(df[col].iloc[0], str) and df[col].iloc[0].isdigit():
            df[col] = df[col].astype(float)
        # Remove trailing zeros
        if isinstance(df[col].iloc[0], (int, float)):
            if int(df[col].iloc[0]) == float(df[col].iloc[0]):
                df[col] = df[col].astype(int)
    return df


def check_and_adjust_data_types(orig_df, new_df):
    """Check and adjust data types and trailing values."""
    for col in new_df.columns[:-2]:
        if orig_df[col].dtype == np.int64:
            new_df[col] = round(new_df[col], 0).astype(int)
        elif orig_df[col].dtype == np.float64:
            # Handle trailing values for float columns
            dec = str(orig_df[col].values[0])[::-1].find('.')
            new_df[col] = round(new_df[col], int(dec))
    return new_df


class PrivateSMOTE:
    """Apply PrivateSMOTE"""

    def __init__(self, samples, N, knn, epsilon, k, key_vars):
        """Initiate arguments"""
        self.samples = samples.reset_index(drop=True)
        self.N = int(N)
        self.knn = knn
        self.epsilon = epsilon
        self.k = k
        self.key_vars = key_vars
        self.newindex = 0
        self.synthetic = None
        self.is_object_type = None
        self.neighbors = None
        self.min_values = None
        self.max_values = None
        self.std_values = None
        self.label_encoders = {}
        self.object_columns = []
        self.unique_values = {}
        self.X_train_shape = ()
        self.x = ()

        # Target variable values
        self.y = np.array(self.samples.loc[:, self.samples.columns[-1]])

        # Create CuPy vector with bool values, where true corresponds to object dtype
        self.is_object_type = np.array(
            self.samples[self.samples.columns[:-1]].dtypes == 'object')

    def khighest_risk(self):
        """Create highest_risk variable based on k-anonymity"""
        kgrp = self.samples.groupby(self.key_vars)[
            self.key_vars[0]].transform(len)
        self.samples['highest_risk'] = np.where(kgrp < self.k, 1, 0)
        return self.samples

    def nearest_neighbours(self, df):
        """Find nearest neighbors using standardized data."""
        self.standardized_data = StandardScaler().fit_transform(df)
        nearestn = NearestNeighbors(
            n_neighbors=self.knn + 1).fit(self.standardized_data)
        return nearestn

    def enc_data(self):
        """Encode categorical data and perform one-hot encoding if necessary."""
        if np.any(self.is_object_type):
            self.label_encoders = {}
            self.label_encoder = LabelEncoder()

            enc_samples = self.samples.loc[:, self.samples.columns[:-2]].copy()

            # Get the column names that are of object type
            self.object_columns = enc_samples.columns[self.is_object_type]
            self.unique_values = {}

            # One-hot encode categorical columns for knn
            dummie_data = np.array(pd.get_dummies(
                enc_samples, drop_first=True))
            self.neighbors = self.nearest_neighbours(dummie_data)

            # Label encode the object-type columns
            enc_samples = np.array(
                self.encode_categorical_columns(enc_samples))

            return enc_samples

        else:
            # Drop highest_risk and target variables to knn
            self.neighbors = self.nearest_neighbours(
                np.array(self.samples.loc[:, self.samples.columns[:-2]]))
            return np.array(self.samples.loc[:, self.samples.columns[:-2]])

    def encode_categorical_columns(self, df):
        """Encode categorical columns."""
        for col in self.object_columns:
            label_encoder = LabelEncoder()
            df[col] = np.array(label_encoder.fit_transform(df[col]))
            self.label_encoders[col] = label_encoder
            self.unique_values[col] = np.unique(df[col])
        return df

    def decode_categorical_columns(self, encoded_data):
        """Decode categorical columns."""
        for col_name in self.object_columns:
            encoded_data[col_name] = self.label_encoders[col_name].inverse_transform(
                encoded_data[col_name].astype(int))
        return encoded_data

    def over_sampling(self):
        """Find the nearest neighbors and populate with new data"""
        N = int(self.N)
        # find highest-risk cases
        self.samples = self.khighest_risk()
        # Highest risk samples selection to be replaced
        self.X_train_shape = self.samples.loc[self.samples['highest_risk']
                                              == 1, self.samples.columns[:-2]].shape
        # Initialize the synthetic samples with the number of samples and attributes
        self.synthetic = np.empty(
            shape=(self.X_train_shape[0] * N, self.X_train_shape[1] + 1), dtype='float32')
        print("all sample: ", self.samples.shape)
        print("n highest risk: ", self.X_train_shape)
        self.x = self.enc_data()
        # Find the minimum value for each numerical column
        self.min_values = [self.x[:, i].min(
        ) if not self.is_object_type[i] else np.nan for i in range(self.x.shape[1])]
        # Find the maximum value for each numerical column
        self.max_values = [self.x[:, i].max(
        ) if not self.is_object_type[i] else np.nan for i in range(self.x.shape[1])]
        # Find the standard deviation value for each numerical column
        self.std_values = [np.std(self.x[:, i]) if not self.is_object_type[i]
                           else np.nan for i in range(self.x.shape[1])]

        # Get the indices of observations that need oversampling
        highest_risk_indices = self.samples.loc[self.samples['highest_risk']
                                                == 1, self.samples.columns[:-2]].index

        # For each observation find nearest neighbours
        for i, _ in enumerate(self.standardized_data):
            if i in highest_risk_indices:
                nnarray = self.neighbors.kneighbors(self.standardized_data[i].reshape(1, -1),
                                                    return_distance=False)[0]
                self._populate(N, i, nnarray)
        
        # assign highest-risk bool value
        highest_risk_col = np.ones((self.synthetic.shape[0], 1), dtype=self.synthetic.dtype)
        new = np.concatenate((self.synthetic, highest_risk_col), axis=1)

        # Convert synthetic data back to a Pandas DataFrame
        new = pd.DataFrame(new, index=range(new.shape[0]),
                           columns=self.samples.columns)

        new = new.astype(dtype=self.samples.dtypes)

        if np.any(self.is_object_type):
            new = self.decode_categorical_columns(new)
        
        # Concatenate highest and non-highest-risk samples
        if self.X_train_shape[0] != self.samples.shape[0]:
            new = pd.concat([new, self.samples.loc[
                self.samples['highest_risk']== 0]])

        return new

    def _populate(self, N, i, nnarray):
        # Populate N times
        while N != 0:
            # Find index of nearest neighbour excluding the observation in comparison
            neighbour = np.random.randint(1, self.knn + 1)

            # Control flip (with standard deviation) for equal neighbor and original values
            flip = [np.multiply(np.multiply(
                np.random.choice([-1, 1], size=1), std),
                np.random.laplace(0, 1 / self.epsilon, size=None))
                if not np.isnan(std)
                else orig_val
                for std, orig_val in zip(self.std_values, self.x[i])]

            # Without flip when neighbour is different from original
            noise = [np.multiply(
                neighbor_val - orig_val,
                np.random.laplace(0, 1 / self.epsilon, size=None))
                if not np.isnan(self.min_values[j])
                else orig_val
                for j, (neighbor_val, orig_val) in enumerate(zip(self.x[nnarray[neighbour]], self.x[i]))]

            # Generate new numerical value for each column
            new_nums_values = [orig_val + flip[j]
                               if neighbor_val == orig_val
                               and not np.isnan(self.min_values[j])
                               and self.min_values[j] <= orig_val + flip[j] <= self.max_values[j]
                               else orig_val - flip[j]
                               if neighbor_val == orig_val and not np.isnan(self.min_values[j])
                               and (self.min_values[j] > orig_val + flip[j]
                                    or orig_val + flip[j] > self.max_values[j])
                               else orig_val + noise[j]
                               if neighbor_val != orig_val and not np.isnan(self.min_values[j])
                               and self.min_values[j] <= orig_val + noise[j] <= self.max_values[j]
                               else orig_val - noise[j]
                               if neighbor_val != orig_val and not np.isnan(self.min_values[j])
                               and (self.min_values[j] > orig_val + noise[j] > self.max_values[j])
                               else orig_val
                               for j, (neighbor_val, orig_val) in enumerate(zip(self.x[nnarray[neighbour]], self.x[i]))]

            # Replace the old categories if there are categorical columns
            if np.any(self.is_object_type):
                nn_unique = [np.unique(self.x[nnarray[1: self.knn + 1], col])
                             for col in np.where(self.is_object_type)[0]]

                # randomly select a category from all existent categories if there is just one category in nn_unique else select from nn_unique
                new_cats_values = [np.random.choice(list(self.unique_values.values())[u], size=1) if len(
                    nn_unique[u]) == 1 else np.random.choice(nn_unique[u], size=1) for u in range(len(self.unique_values))]

                # replace the old categories
                iter_cat_values = iter(new_cats_values)

                new_nums_values = [next(iter_cat_values) if np.isnan(
                    self.min_values[j]) else val for j, val in enumerate(new_nums_values)]

            # Concatenate the arrays along axis=0
            synthetic_array = np.hstack(new_nums_values)

            # Assign interpolated values
            self.synthetic[self.newindex,
                           0: synthetic_array.shape[0]] = synthetic_array

            # Assign intact target variable
            self.synthetic[self.newindex, synthetic_array.shape[0]] = self.y[i]
            self.newindex += 1
            N -= 1


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Apply interpolation with PrivateSMOTE')
    parser.add_argument('--input_file', type=str, default="none",
                        required=True, help='Path of the input file')
    parser.add_argument('--knn', type=int, default=1,
                        help='Nearest Neighbor for interpolation')
    parser.add_argument('--per', type=int, default=1,
                        help='Amount of new cases to replace the original')
    parser.add_argument('--epsilon', type=float,
                        default=0.5, help='Amount of noise')
    parser.add_argument('--k', type=int, default=2,
                        help='Group size for k-anonymity')
    parser.add_argument('--key_vars', nargs='+', default=[],
                        required=True, help='Quasi-Identifiers')
    args = parser.parse_args()

    # Read data
    data = pd.read_csv(f'{args.input_file}')

    # encode string with numbers to numeric and remove trailing zeros
    data = keep_numbers(data)

    # encoded target
    tgt_obj = data[data.columns[-1]].dtypes == 'object'
    if tgt_obj:
        target_encoder = LabelEncoder()
        data[data.columns[-1]
             ] = target_encoder.fit_transform(data[data.columns[-1]])

    # Apply PrivateSMOTE
    newDf = PrivateSMOTE(data, args.per, args.knn, args.epsilon,
                         args.k, args.key_vars).over_sampling()

    if tgt_obj:
        newDf[newDf.columns[-2]
              ] = target_encoder.inverse_transform(newDf[newDf.columns[-2]])

    # Check and adjust data types and trailing values
    newDf = check_and_adjust_data_types(data, newDf)

    # Save synthetic data
    newDf.to_csv(
        f'synth_data{sep}{args.input_file.split(".csv")[0]}_{args.epsilon}-privateSMOTE_QI0_knn{args.knn}_per{args.per}.csv', index=False)
