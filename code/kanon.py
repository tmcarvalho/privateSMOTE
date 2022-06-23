"""Find singleouts to synthetise
This script will evaluate the single out cases.
"""

import random
import pandas as pd
import numpy as np
from collections import Counter

def single_outs_sets(data):
    """It takes a dataframe and returns a new dataframe with a new column called 'single_out'
    that is 1 if the row is a single out and 0 otherwise, based on select quasi-identifiers

    :param data: the dataframe you want to create the single out variable

    Returns:
        dataframe: Dataframe with single outs variable
        list: selected quasi-identifiers
    """
    set_data = []
    set_key_vars = []
    key_vars_wrep = []
    # select 10% of attributes as quasi-identifiers
    random.seed(42)
    for i in range(0, 7):
        # change threshold of selected quasi-identifers based on number of columns
        key_vars = random.sample(
            sorted(data.columns[:-1]),
            k=int(round(0.25*len(data.columns), 0)))\
            if len(data.columns) >= 10\
                else random.sample(sorted(
                    data.columns[:-1]),
                    k=int(round(0.4*len(data.columns), 0)))
        key_vars.sort()
        set_key_vars.append(key_vars)
        [key_vars_wrep.append(x) for x in set_key_vars if x not in key_vars_wrep]

    if len(key_vars_wrep) > 5:
        key_vars_wrep = key_vars_wrep[:5]

    for key_vars in key_vars_wrep:
        k = data.groupby(key_vars)[key_vars[0]].transform(len)

        data_copy = data.copy()
        data_copy['single_out'] = None
        data_copy['single_out'] = np.where(k == 1, 1, 0)

        set_data.append(data_copy)

    return set_data, key_vars_wrep


def kanonymity(obj):
    """
    Measure the individual risk base on the frequency of the equivalence classes.
    :param obj: input data set.
    :return: frequencies of equivalence classes.
    """

    keyVars = list(obj.columns.values)
    # get the frequencies of equivalence classes for each observation
    fk = obj.groupby(keyVars).size()
    # group the frequencies
    fk_grp = Counter(fk)

    return fk_grp
