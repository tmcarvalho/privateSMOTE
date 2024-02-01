"""Apply interpolation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
# Import necessary libraries
from os import sep
import re
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_file', type=str, default="none")
parser.add_argument('--key_vars', nargs='+', default=[])
parser.add_argument('--knn', type=int, default="none")
parser.add_argument('--per', type=int, default="none")
parser.add_argument('--epsilon', type=float, default="none")
args = parser.parse_args()

def keep_numbers(data):
    """Fix data types according to the data"""
    for col in data.columns:
        # Transform numerical strings to digits
        if isinstance(data[col].iloc[0], str) and data[col].iloc[0].isdigit():
            data[col] = data[col].astype(float)
        # Remove trailing zeros
        if isinstance(data[col].iloc[0], (int, float)):
            if int(data[col].iloc[0]) == float(data[col].iloc[0]):
                data[col] = data[col].astype(int)
    return data

def aux_singleouts(key_vars, dt):
    """Create single out variable based on k-anonymity"""
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = np.where(k < 3, 1, 0)
    return dt

def check_and_adjust_data_types(origDF, newDf):
    for col in newDf.columns[:-2]:
        if origDF[col].dtype == np.int64:
            newDf[col] = round(newDf[col], 0).astype(int)
        elif origDF[col].dtype == np.float64:
            # Handle trailing values for float columns
            dec = str(origDF[col].values[0])[::-1].find('.')
            newDf[col] = round(newDf[col], int(dec))
    return newDf

class PrivateSMOTE:
    """Apply PrivateSMOTE"""
    def __init__(self, samples, N, k, ep):
        """Initiate arguments"""
        self.samples = samples.reset_index(drop=True)
        self.N = int(N)
        self.k = k
        self.ep = ep
        self.newindex = 0

        # Single out samples that will be replaced
        self.X_train_shape = self.samples.loc[self.samples['single_out'] == 1, self.samples.columns[:-2]].shape
        print("n singleouts: ", self.X_train_shape)
        # Target variable values
        self.y = np.array(self.samples.loc[:, self.samples.columns[-2]])

        # Initialize the synthetic samples with the number of samples and attributes
        self.synthetic = np.empty(shape=(self.X_train_shape[0] * N, self.X_train_shape[1] + 1), dtype='float32')

        # Create CuPy vector with bool values, where true corresponds to object dtype
        self.is_object_type = np.array(self.samples[self.samples.columns[:-2]].dtypes == 'object')

        print(set(self.samples[self.samples.columns[:-2]].dtypes))


    def enc_data(self):
        if np.any(self.is_object_type):
            self.label_encoders = {}
            self.label_encoder = LabelEncoder()
            enc_samples = self.samples.loc[:,self.samples.columns[:-2]].copy()

            # Get the column names that are of object type
            self.object_columns = enc_samples.columns[self.is_object_type]
            self.unique_values = {}
            
            # One-hot encode categorical columns for knn
            dummie_data = np.array(pd.get_dummies(enc_samples, drop_first=True))
            self.neighbors = self.nearest_neighbours(dummie_data)

            # Label encode the object-type columns
            enc_samples = np.array(self.encode_categorical_columns(enc_samples))

            return enc_samples
        
        else:
            # Drop singlout and target variables to knn
            self.neighbors = self.nearest_neighbours(np.array(self.samples.loc[:, self.samples.columns[:-2]]))
            return np.array(self.samples.loc[:, self.samples.columns[:-2]])

    def encode_categorical_columns(self, data):
        for col in self.object_columns:
            label_encoder = LabelEncoder()
            data[col] = np.array(label_encoder.fit_transform(data[col]))
            self.label_encoders[col] = label_encoder
            self.unique_values[col] = np.unique(data[col])
        return data

    def decode_categorical_columns(self, encoded_data):
        # Inverse transform the column
        for col_name in self.object_columns:
            encoded_data[col_name] = self.label_encoders[col_name].inverse_transform(encoded_data[col_name].astype(int))
        return encoded_data
    
    def nearest_neighbours(self, data):
        # Standardize the data
        self.standardized_data = StandardScaler().fit_transform(data)
        nearestn = NearestNeighbors(n_neighbors=self.k + 1).fit(self.standardized_data)
        return nearestn

    def over_sampling(self):
        """Find the nearest neighbors and populate with new data"""
        N = int(self.N)
        self.x = self.enc_data()
        # Find the minimum value for each numerical column
        self.min_values = [self.x[:, i].min() if not self.is_object_type[i] else np.nan for i in range(self.x.shape[1])]
        # Find the maximum value for each numerical column
        self.max_values = [self.x[:, i].max() if not self.is_object_type[i] else np.nan for i in range(self.x.shape[1])]
        # Find the standard deviation value for each numerical column
        self.std_values = [np.std(self.x[:, i]) if not self.is_object_type[i] else np.nan for i in range(self.x.shape[1])]

        # Get the indices of observations that need oversampling
        single_out_indices = self.samples.loc[self.samples['single_out'] == 1, self.samples.columns[:-2]].index

        # For each observation find nearest neighbours
        for i, _ in enumerate(self.standardized_data):
            if i in single_out_indices:
                nnarray = self.neighbors.kneighbors(self.standardized_data[i].reshape(1, -1),
                                               return_distance=False)[0]
                
                self._populate(N, i, nnarray)

        self.synthetic = self.synthetic

        # Convert synthetic data back to a Pandas DataFrame
        new = pd.DataFrame(self.synthetic, columns=self.samples.columns[:-1])
        new = new.astype(dtype=self.samples[self.samples.columns[:-1]].dtypes)
        if np.any(self.is_object_type):
            new = self.decode_categorical_columns(new)
        return new

    def _populate(self, N, i, nnarray):
        # Populate N times
        while N != 0:
            # Find index of nearest neighbour excluding the observation in comparison
            neighbour = np.random.randint(1, self.k + 1)

            # Control flip (with standard deviation) for equal neighbor and original values 
            flip = [np.multiply(np.multiply(
                            np.random.choice([-1, 1], size=1), std),
                            np.random.laplace(0, 1 / self.ep, size=None))
                            if not np.isnan(std)
                            else orig_val
                            for std, orig_val in zip(self.std_values, self.x[i])]

            # Without flip when neighbour is different from original
            noise = [np.multiply(
                            neighbor_val - orig_val,
                            np.random.laplace(0, 1 / self.ep, size=None))
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
                nn_unique = [np.unique(self.x[nnarray[1 : self.k + 1], col]) 
                            for col in np.where(self.is_object_type)[0]]
                
                # randomly select a category from all existent categories if there is just one category in nn_unique else select from nn_unique
                new_cats_values = [np.random.choice(list(self.unique_values.values())[u], size=1) if len(nn_unique[u]) == 1 else np.random.choice(nn_unique[u], size=1) for u in range(len(self.unique_values))]

                # replace the old categories
                iter_cat_values = iter(new_cats_values)

                new_nums_values = [next(iter_cat_values) if np.isnan(self.min_values[j]) else val for j, val in enumerate(new_nums_values)]


            # Concatenate the arrays along axis=0
            synthetic_array = np.hstack(new_nums_values)
                    
            # Assign interpolated values
            self.synthetic[self.newindex, 0 : synthetic_array.shape[0]] = synthetic_array

            # Assign intact target variable
            self.synthetic[self.newindex, synthetic_array.shape[0]] = self.y[i]
            self.newindex += 1
            N -= 1


# %% 
data = pd.read_csv(f'{args.input_file}')

# encode string with numbers to numeric and remove trailing zeros
data = keep_numbers(data)

data = aux_singleouts(args.key_vars, data)

# encoded target
tgt_obj = data[data.columns[-2]].dtypes == 'object'
if tgt_obj:
    target_encoder = LabelEncoder()
    data[data.columns[-2]] = target_encoder.fit_transform(data[data.columns[-2]])


newDf = PrivateSMOTE(data, args.per, args.knn, args.epsilon).over_sampling()

# Convert synthetic data back to a Pandas DataFrame
newDf = pd.concat([newDf, pd.DataFrame({'single_out': [1] * newDf.shape[0]})], axis=1)
if newDf.shape[0] != data.shape[0]:
    newDf = pd.concat([newDf, data.loc[data['single_out'] == 0]])

if tgt_obj:
    newDf[newDf.columns[-2]] = target_encoder.inverse_transform(newDf[newDf.columns[-2]])


# Check and adjust data types and trailing values
newDf = check_and_adjust_data_types(data, newDf)

newDf.to_csv(f'synth_data{sep}{args.input_file.split(".csv")[0]}_{args.epsilon}-privateSMOTE_QI0_knn{args.knn}_per{args.per}.csv', index=False)
