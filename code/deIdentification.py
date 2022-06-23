"""Data de-identification
This script will de-identify the data for 5 set of quasi-identifiers.
"""
# %%
import pandas as pd
import transformations
from os import sep, walk
from kanon import single_outs_sets

def apply_transformations(obj, key_vars, tech_comb, parameters, result):
    """Apply transformations

    Args:
        obj (pd.Dataframe): dataframe for de-identification
        key_vars (list): list of quasi-identifiers
        tech_comb (list): combination of techniques
        parameters (list): parameters of techniques
        result (dictionary): dictionary to store results

    Returns:
        tuple: last transformed variant, dictionary of results
    """

    if 'sup' in tech_comb:
        param = parameters[[tech_comb.index(l) for l in tech_comb if 'sup'==l][0]] if len(tech_comb)>1 else parameters
        obj = transformations.suppression(obj, key_vars, uniq_per=param)    
    if 'topbot' in tech_comb:
        param = parameters[[tech_comb.index(l) for l in tech_comb if 'topbot'==l][0]] if len(tech_comb)>1 else parameters
        obj = transformations.topBottomCoding(obj, key_vars, outlier=param)
    if 'round' in tech_comb:
        param = parameters[[tech_comb.index(l) for l in tech_comb if 'round'==l][0]] if len(tech_comb)>1 else parameters
        obj = transformations.rounding(obj, key_vars, base=param)
    if 'globalrec' in tech_comb:
        param = parameters[[tech_comb.index(l) for l in tech_comb if 'globalrec'==l][0]] if len(tech_comb)>1 else parameters
        obj = transformations.globalRecoding(obj, key_vars, std_magnitude=param)

    # if transformed variant is different from others
    if (len(result['combination'])==0) or (not(any(x.equals(obj) for x in result['transformedDS'])) and len(result['combination'])!=0):
        result['combination'].append(tech_comb)  
        result['parameters'].append(parameters) 
        result['key_vars'].append(key_vars)
        result['transformedDS'].append(obj)

    return obj, result


def process_transformations(df, key_vars):
    """Find combinations, respective parameterisation and apply transformations.

    Args:
        df (pd.Dataframe): dataframe for de-identification
        key_vars (list): list of quasi-identifiers

    Returns:
        dictionary: set transformed variants
    """

    df_val = df.copy()

    # create combinations adequate to the data set
    comb_name, param_comb = transformations.parameters(df_val, key_vars)

    result = {'combination': [], 'parameters': [], 'key_vars': [], 'transformedDS': []}

    # transform data
    df_transf = df_val.copy()
    if len(param_comb) > 1:
        for comb in param_comb:
            # apply transformations
            df_transf, result = apply_transformations(df_transf, key_vars, comb_name, comb, result)
    else:
        # apply transformations
        for i in param_comb[0]:
            df_transf, result = apply_transformations(df_transf, key_vars, comb_name, i, result)            

    return result


# %% save best transformed variant for each combined technique
# path to input data
input_folder = '../original/'
transformed_folder = '../PPT'
risk_folder = '../output/record_linkage/PPT'

_, _, input_files = next(walk(f'{input_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87]
set_qis = {'ds':[], 'set_key_vars':[]}

for idx, file in enumerate(input_files):
    
    if int(file.split(".csv")[0]) not in not_considered_files:
        df =  pd.read_csv(f'{input_folder}/{file}')
        # get index
        file_idx = int(file.split('.')[0])
        _, set_key_vars = single_outs_sets(df)
        
        if len(set_key_vars) == 5:
            print(idx)
            set_qis['ds'].append(file_idx)
            set_qis['set_key_vars'].append(set_key_vars)
            
            for j, key_vars in enumerate(set_key_vars):
                # apply de-identification to the set of key vars    
                result = process_transformations(df, key_vars)
                # transform dict to dataframe
                res_df = pd.DataFrame(result)

                for i in range(len(res_df)):
                    res_df['transformedDS'][i].to_csv(f'{transformed_folder}{sep}ds{str(file_idx)}_transf{str(i)}_qi{j}.csv', index=False)

# %%
set_qis_df = pd.DataFrame(set_qis)
set_qis_df.to_csv(f'{transformed_folder}{sep}list_key_vars.csv', index=False)

# %%
