# %%
from os import sep, walk
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from kanon import single_outs_sets
import random
from sklearn.neighbors import NearestNeighbors
# %%
#################### SMOTE FROM SCRATCH #######################
import random
from random import randrange
from sklearn.neighbors import NearestNeighbors

class Smote:
    def __init__(self,samples,y,N,k):
        """Initiate arguments

        Args:
            samples (array): training samples
            y (1D array): target sample
            N (int): number of interpolations per observation
            k (int): number of nearest neighbours
        """
        self.n_samples = samples.shape[0]
        self.n_attrs=samples.shape[1]
        self.y=y
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0

    def over_sampling(self):
        N=int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs+1))
        neighbors=NearestNeighbors(n_neighbors=self.k+1).fit(self.samples)

        # for each observation find nearest neighbours
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)

        return self.synthetic

    def _populate(self,N,i,nnarray):
        # populate N times
        for j in range(N):
            # find index of nearest neighbour excluding the observation in comparison
            neighbour = randrange(1, self.k+1)

            difference = abs(self.samples[i]-self.samples[nnarray[neighbour]])
            # multiply with a weight
            weight = random.uniform(0, 1)
            additive = np.multiply(difference,weight)

            # assign interpolated values
            self.synthetic[self.newindex, 0:len(self.synthetic[self.newindex])-1] = self.samples[i]+additive
            # assign intact target variable
            self.synthetic[self.newindex, len(self.synthetic[self.newindex])-1] = self.y[i]
            self.newindex+=1

# %%
def interpolation_singleouts_scratch(original_folder, file):
    """Generate several interpolated data sets considering all classes.

    Args:
        original_folder (string): path of original folder
        file (string): name of file
    """

    output_interpolation_folder = '../output/oversampled/smote_singleouts_scratch/'
    data = pd.read_csv(f'{original_folder}/{file}')

    # apply LabelEncoder beacause of smote
    label_encoder_dict = defaultdict(LabelEncoder)
    data_encoded = data.apply(lambda x: label_encoder_dict[x.name].fit_transform(x))
    map_dict = dict()
    
    for k in data.columns:
        if data[k].dtype=='object':
            keys = data[k]
            values = data_encoded[k]
            sub_dict = dict(zip(keys, values))
            map_dict[k] = sub_dict

    set_data, _ = single_outs_sets(data_encoded)

    for idx, dt in enumerate(set_data):
        X_train = dt.loc[dt['single_out']==1, dt.columns[:-2]]
        Y_train = dt.loc[dt['single_out']==1, dt.columns[-1]]
        y = dt.loc[dt['single_out']==1, dt.columns[-2]]

        # getting the number of singleouts in training set
        singleouts = Y_train.shape[0]

        # storing the singleouts instances separately
        x1 = np.ones((singleouts, X_train.shape[1]))
        x1=[X_train.iloc[i] for i, v in enumerate(Y_train) if v==1.0]
        x1=np.array(x1)

        y=np.array(y)

        knn = [1,3,5]
        per = [1,2,3]
        for k in knn:
            for p in per:
                try:
                    new = Smote(x1, y, p, k).over_sampling()
                    newDf = pd.DataFrame(new)
                    # restore feature name 
                    newDf.columns = dt.columns[:-1]
                    # assign singleout
                    newDf[dt.columns[-1]] = 1
                    # add non single outs
                    newDf = pd.concat([newDf, dt.loc[dt['single_out']==0]])
                    for col in newDf.columns:
                        if dt[col].dtype == np.int64:
                            newDf[col] = round(newDf[col], 0).astype(int)
                        else:    
                            newDf[col] = newDf[col].astype(dt[col].dtype)
                    
                    # decoded
                    for key in map_dict.keys():
                        d = dict(map(reversed, map_dict[key].items()))
                        newDf[key] = newDf[key].map(d)

                    # save oversampled data
                    newDf.to_csv(
                        f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_smote_QI{idx}_knn{k}_per{p}.csv',
                        index=False)

                except:
                    pass
# %%
original_folder = '../original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        interpolation_singleouts_scratch(original_folder, file)
