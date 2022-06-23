""" 
This script will modeling data
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from modeling import evaluate_model
import re


def save_results(file, args, validation, test):
    """Create a folder if dooes't exist and save results

    Args:
        file (string): file name
        args (args): command line arguments
        validation (dictionary): validation of training results
        test (dictionary): test results
    """
    output_folder_val = (
        f'{args.output_folder}/validation')
    output_folder_test = (
        f'{args.output_folder}/test')
    if not os.path.exists(output_folder_val) |\
            os.path.exists(output_folder_test):
        os.makedirs(output_folder_val)
        os.makedirs(output_folder_test)
    np.save(
        f'{output_folder_val}/{file.replace("csv", "npy")}', validation)
    np.save(
        f'{output_folder_test}/{file.replace("csv", "npy")}', test)


def modeling_ppt(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file

    Raises:
        Exception: failed to apply smote when single outs
        class is great than non single outs.
        exc: failed to writing the results.
    """
    
    print(f'{args.input_folder}/{file}')

    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

    data = pd.read_csv(f'{args.input_folder}/{file}')

    # prepare data to modeling
    data = data.apply(LabelEncoder().fit_transform)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    # split data 80/20
    train_idx = list(set(list(X.index)) - set(index))
    x_train = X.iloc[train_idx, :]
    x_test = X.iloc[index, :]
    y_train = y[train_idx]
    y_test = y[index]

    # predictive performance
    validation, test = evaluate_model(x_train, x_test, y_train, y_test)

    # save validation and test results
    try:
       save_results(file, args, validation, test)

    except Exception as exc:
        raise exc



def modeling_smote_under_over(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file

    Raises:
        Exception: failed to apply smote when single outs
        class is great than non single outs.
        exc: failed to writing the results.
    """
    
    print(f'{args.input_folder}/{file}')

    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

    data = pd.read_csv(f'{args.input_folder}/{file}')
    print(f)
    if 'under' in file:
        orig_folder = 'original'
        _, _, orig_files = next(os.walk(f'{orig_folder}'))
 
        orig_file = [file for file in orig_files if list(map(int, re.findall(r'\d+', file.split('_')[0])))[0] == f[0]]
        
        orig_data = pd.read_csv(f'{orig_folder}/{orig_file[0]}')
        test_data = orig_data.iloc[index,:]
        print(len(data))
        print(len(test_data))
        dups = pd.concat([data, test_data])
        dups = dups[dups.duplicated(keep='first')]
        print(len(dups))
        if len(dups) != 0:
            dups_index = list(dups.index)
            index = list(set(index)-set(dups_index))
            print(len(index))
        # prepare data to modeling
        data = data.apply(LabelEncoder().fit_transform)
        orig_data = orig_data.apply(LabelEncoder().fit_transform)

        x_train, y_train = data.iloc[:, :-1], data.iloc[:, -1]
        x_test = orig_data.iloc[index, :-1]
        y_test = orig_data.iloc[index, -1]
    
    else:
        print(len(index))
        # prepare data to modeling
        data = data.apply(LabelEncoder().fit_transform)

        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        # split data 80/20
        train_idx = list(set(list(X.index)) - set(index))
        x_train = X.iloc[train_idx, :]
        x_test = X.iloc[index, :]
        y_train = y[train_idx]
        y_test = y[index]

    # predictive performance
    validation, test = evaluate_model(x_train, x_test, y_train, y_test)

    # save validation and test results
    try:
       save_results(file, args, validation, test)

    except Exception as exc:
        raise exc



def modeling_singleouts(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file

    Raises:
        Exception: failed to apply smote when single outs
        class is great than non single outs.
        exc: failed to writing the results.
    """
    print(f'{args.input_folder}/{file}')

    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

    orig_folder = 'original'
    _, _, orig_files = next(os.walk(f'{orig_folder}'))
    orig_file = [file for file in orig_files if list(map(int, re.findall(r'\d+', file.split('_')[0])))[0] == f[0]]
    print(orig_file)
    orig_data = pd.read_csv(f'{orig_folder}/{orig_file[0]}')
    data = pd.read_csv(f'{args.input_folder}/{file}')

    # prepare data to modeling
    orig_data = orig_data.apply(LabelEncoder().fit_transform)
    data = data.apply(LabelEncoder().fit_transform)
    x_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]

    x_test = orig_data.iloc[index, :-1]
    y_test = orig_data.iloc[index, -1]

    if y_train.value_counts().nunique() != 1:
        # predictive performance
        validation, test = evaluate_model(x_train, x_test, y_train, y_test)
        # save validation and test results
        try:
            if validation:
                save_results(file, args, validation, test)

        except Exception as exc:
            raise exc

# %%
