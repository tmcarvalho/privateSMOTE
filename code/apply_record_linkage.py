"""_summary_
"""
# %%
from io import StringIO
from os import walk
import pandas as pd
import numpy as np
from record_linkage import threshold_record_linkage
from sklearn.preprocessing import LabelEncoder
import re
import gc
import ast
import zipfile

# %%
def privacy_risk(transf_file, orig_data, args, list_key_vars):
    dict_per = {'privacy_risk_50': [], 'privacy_risk_70': [], 'privacy_risk_90':[], 'privacy_risk_100': [], 'ds': []}

    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    int_transf_qi = list(map(int, re.findall(r'\d+', transf_file.split('_')[2])))
    set_key_vars = list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0]
    print(transf_file) 
    print(int_transf_files)

    transf_data = pd.read_csv(f'{args.input_folder}/{transf_file}')
    if args.type == 'smote_singleouts':
        transf_data = transf_data[transf_data['single_out']==1]
    
    # apply LabelEncoder for modeling
    orig_data = orig_data.apply(LabelEncoder().fit_transform)
    transf_data = transf_data.apply(LabelEncoder().fit_transform)

    if int_transf_qi[0] == 0:
        key_vars = ast.literal_eval(set_key_vars)[0]
        if args.type == 'smote_singleouts':
            orig_data = aux_singleouts(key_vars, orig_data)
        
        key_vars = [k for k in key_vars if transf_data[k].values[0]!='*']
        matches, percentages = threshold_record_linkage(
            transf_data,
            orig_data,
            key_vars)

    if int_transf_qi[0] == 1:
        key_vars = ast.literal_eval(set_key_vars)[1]
        if args.type == 'smote_singleouts':
            orig_data = aux_singleouts(key_vars, orig_data)

        key_vars = [k for k in key_vars if transf_data[k].values[0]!='*']
        matches, percentages = threshold_record_linkage(
            transf_data,
            orig_data,
            key_vars)
    
    if int_transf_qi[0] == 2:
        key_vars = ast.literal_eval(set_key_vars)[2]
        if args.type == 'smote_singleouts':
            orig_data = aux_singleouts(key_vars, orig_data)

        key_vars = [k for k in key_vars if transf_data[k].values[0]!='*']
        matches, percentages = threshold_record_linkage(
            transf_data,
            orig_data,
            key_vars)
    
    if int_transf_qi[0] == 3:
        key_vars = ast.literal_eval(set_key_vars)[3]
        if args.type == 'smote_singleouts':
            orig_data = aux_singleouts(key_vars, orig_data)

        key_vars = [k for k in key_vars if transf_data[k].values[0]!='*']
        matches, percentages = threshold_record_linkage(
            transf_data,
            orig_data,
            key_vars)
    
    if int_transf_qi[0] == 4:
        key_vars = ast.literal_eval(set_key_vars)[4]
        if args.type == 'smote_singleouts':
            orig_data = aux_singleouts(key_vars, orig_data)

        key_vars = [k for k in key_vars if transf_data[k].values[0]!='*']
        matches, percentages = threshold_record_linkage(
            transf_data,
            orig_data,
            key_vars)
    
    with zipfile.ZipFile(f'{args.output_folder}/potential_matches.zip', "a", zipfile.ZIP_DEFLATED) as zip_file:
        s = StringIO()
        matches.to_csv(s, index=False) 
        zip_file.writestr(f'{transf_file.split(".csv")[0]}_rl.csv', s.getvalue())
    # matches.to_csv(f'{args.output_folder}/{transf_file.split(".csv")[0]}_rl.csv', index=False) 
    dict_per['privacy_risk_50'].append(percentages[0])
    dict_per['privacy_risk_70'].append(percentages[1])
    dict_per['privacy_risk_90'].append(percentages[2])
    dict_per['privacy_risk_100'].append(percentages[3])
    dict_per['ds'].append(transf_file.split('.csv')[0])
    gc.collect()
    
    return dict_per


def aux_singleouts(key_vars, dt):
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = None
    dt['single_out'] = np.where(k == 1, 1, 0)
    dt = dt[dt['single_out']==1]
    return dt


def privacy_risk_smote_under_over(transf_file, orig_data, args, key_vars, i):
    dict_per = {'privacy_risk_50': [], 'privacy_risk_70': [], 'privacy_risk_90':[], 'privacy_risk_100': [], 'ds': []}
    
    transf_data = pd.read_csv(f'{args.input_folder}/{transf_file}')
        
    # apply LabelEncoder for modeling
    orig_data = orig_data.apply(LabelEncoder().fit_transform)
    transf_data = transf_data.apply(LabelEncoder().fit_transform)
    print(transf_file)

    matches, percentages = threshold_record_linkage(
        transf_data,
        orig_data,
        key_vars)
    
    with zipfile.ZipFile(f'{args.output_folder}/potential_matches.zip', "a", zipfile.ZIP_DEFLATED) as zip_file:
        s = StringIO()
        matches.to_csv(s, index=False) 
        zip_file.writestr(f'{transf_file.split(".csv")[0]}_qi{i}_rl.csv', s.getvalue())
    # matches.to_csv(f'{args.output_folder}/{transf_file.split(".csv")[0]}_qi{i}_rl.csv', index=False) 
    dict_per['privacy_risk_50'].append(percentages[0])
    dict_per['privacy_risk_70'].append(percentages[1])
    dict_per['privacy_risk_90'].append(percentages[2])
    dict_per['privacy_risk_100'].append(percentages[3])
    dict_per['ds'].append(transf_file.split('.csv')[0])
    dict_per['qi'] = i
    gc.collect()

    return dict_per  


# %% 
def apply_privacy_risk(transf_file, args):
    original_folder = 'original'
    _, _, input_files = next(walk(f'{original_folder}'))

    list_key_vars = pd.read_csv('PPT/list_key_vars.csv')
    
    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))
    orig_file = [file for file in input_files if int(file.split(".csv")[0]) == int_transf_files[0]]
    print(orig_file)
    orig_data = pd.read_csv(f'{original_folder}/{orig_file[0]}')

    risk = privacy_risk(transf_file, orig_data, args, list_key_vars)
    total_risk = pd.DataFrame.from_dict(risk)
    total_risk.to_csv(f'{args.output_folder}/{transf_file.split(".csv")[0]}_per.csv', index=False) 


def apply_in_smote_under_over(transf_file, args):
    original_folder = 'original'
    _, _, input_files = next(walk(f'{original_folder}'))

    int_transf_files = list(map(int, re.findall(r'\d+', transf_file.split('_')[0])))

    list_key_vars = pd.read_csv('PPT/list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==int_transf_files[0], 'set_key_vars'].values[0])

    orig_file = [file for file in input_files if int(file.split(".csv")[0]) == int_transf_files[0]]
    print(int_transf_files)
    orig_data = pd.read_csv(f'{original_folder}/{orig_file[0]}')
    print(orig_file)

    for i in range(len(set_key_vars)):
        risk = privacy_risk_smote_under_over(transf_file, orig_data, args, set_key_vars[i], i)
        total_risk = pd.DataFrame.from_dict(risk)
        total_risk.to_csv(f'{args.output_folder}/{transf_file.split(".csv")[0]}_qi{i}_per.csv', index=False) 
