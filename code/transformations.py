import numpy as np
import pandas as pd
import itertools
import math


def tuckey_method(obj, column, outlier):
    # outliers detection with Tukey's method
    q1 = obj[column].quantile(0.25)
    q3 = obj[column].quantile(0.75)
    iqr = q3 - q1
    outer_fence = outlier * iqr

    # outer fence lower and upper end
    outer_fence_le = q1 - outer_fence
    outer_fence_ue = q3 + outer_fence

    outliers_prob = []
    for index, x in enumerate(obj[column]):
        if (x <= outer_fence_le or x >= outer_fence_ue) and outer_fence_le != outer_fence_ue:
            outliers_prob.append(index)

    return outliers_prob, outer_fence_le, outer_fence_ue

            
def parameters(obj, key_vars):

    sup_parameters = []
    round_parameters = []
    gen_parameters = []
    topbot_parameters = []

    # for suppression
    uniques_per = obj[key_vars].apply(lambda col: col.nunique() / len(obj))
    uniques_per = list(uniques_per)
    uniques_per = [math.floor(i*10)/10 for i in uniques_per]
    # define maximum percentage
    sup_parameters = [j for i in [0.7, 0.8, 0.9] for j in uniques_per if j == i and len(uniques_per)!=0]

    # for rounding
    round_vars = obj[key_vars].select_dtypes(include=np.float64).columns
    if len(round_vars) != 0:
        round_parameters = [0.2, 2, 5, 10]
    
    # for generalisation
    gen_vars = obj[key_vars].select_dtypes(include=np.int).columns
    if len(gen_vars) != 0:
        for col in gen_vars:
            sigma = np.std(obj[col])
            if sigma!=0:
                gen_parameters = [0.5, 1.5 ,3]

    # for top and bottom
    topbot_vars = obj[key_vars].select_dtypes(include=np.number).columns
    topbot_parameters = []
    outliers_prob = []
    for outlier in [1.5, 3]:
        for col in topbot_vars:
            outliers_prob, _, _ = tuckey_method(obj, col, outlier)   
        
        if len(outliers_prob) != 0:
            topbot_parameters.append(outlier)

    output = {'sup': sup_parameters, 'topbot': topbot_parameters, 'round':round_parameters, 'globalrec': gen_parameters}
    
    # remove empty parameters
    output = {i:j for i,j in output.items() if j != []}
    
    if len(output) == 1:
        return list(output.keys()), list(output.values())
    
    else:
        values = [j for _,j in output.items()]
        comb_output = list(itertools.product(*values))

        return list(output.keys()), comb_output



def suppression(obj, key_vars, uniq_per):
    """
    Suppression of columns which have a number of unique values above 90% excluding floating points.
    :param obj: input data set.
    :param uniq_per: percentage of distinct values.
    :return: suppressed data set and the dictionary res.
    """
    uniques_per = obj[key_vars].apply(lambda col: col.nunique() / len(obj))
    uniques_max_per = uniques_per[uniques_per > uniq_per]
    df_sup = obj.copy()
    if len(uniques_max_per) != 0:
        # list of columns to suppress
        cols = df_sup[key_vars].columns[df_sup[key_vars].columns.isin(uniques_max_per.index)].values
        # create key : scalar value dictionary
        scalar_dict = {c: '*' for c in cols}
        # assign columns with '*' which represents the suppression
        df_sup = df_sup.assign(**scalar_dict)

    return df_sup



def rounding(obj, key_vars, base=2):
    """
    Round to specific base.
    :param obj: input data set.
    :param base: rounding base.
    :return: data set with rounded bases and dictionary res.
    """
    round_vars = obj[key_vars].select_dtypes(include=np.float64).columns
    df_round = obj.copy()

    for col in round_vars:
        # guarantee the minimum utility --> base < max(df_round[col])
        df_round[col] = df_round[col].apply(lambda x: base * round(x/base))
        df_round[col] = df_round[col].astype(np.float)
        
    return df_round               



def globalRecoding(obj, key_vars, std_magnitude=1):
    """
    Global recoding of numerical (continuous) variables.
    :param obj: input data set.
    :param std_magnitude: standard deviation magnitude to define the binning size.
    :return: recoded data set and the disctionary res.
    """
    
    gen_vars = obj[key_vars].select_dtypes(include=np.int).columns
    df_gen = obj.copy()

    for col in gen_vars:
        sigma = np.std(obj[col])
        mg = int(sigma * std_magnitude)
        if sigma!=0 and mg!=0:
            bins = list(range(min(df_gen[col]), max(df_gen[col]) + mg, mg))
            labels = ['%d' % bins[i] for i in range(0, len(bins) - 1)]
            df_gen[col] = pd.cut(obj[col], bins=bins, labels=labels, include_lowest=True).astype(int)
        
        else:
            pass

    return df_gen  

    

def topBottomCoding(obj, key_vars, outlier=1.5):
    """
    Replace extreme values, larger or lower than a threshold, by a different value.
    :param obj: input data set.
    :param outlier: inner or outer fence values to find outliers.
    :return: top or bottom coded data.
    """

    topbot_vars = obj[key_vars].select_dtypes(include=np.number).columns
    data_to_transform = obj.copy()

    for col in  topbot_vars:  
        _, outer_fence_le, outer_fence_ue = tuckey_method(obj, col, outlier)

        data_to_transform.loc[
            data_to_transform[col] <= outer_fence_le, col] = outer_fence_le
        data_to_transform.loc[
            data_to_transform[col] >= outer_fence_ue, col] = outer_fence_ue

    return data_to_transform  