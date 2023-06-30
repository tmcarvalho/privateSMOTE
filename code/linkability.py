#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import re
import ast

from anonymeter.evaluators import LinkabilityEvaluator


def anonymeter_linkability(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file
    """
    print(f'{args.input_folder}{file}')

    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

    orig_folder = 'original'
    _, _, orig_files = next(os.walk(f'{orig_folder}'))
    orig_file = [fl for fl in orig_files if list(map(int, re.findall(r'\d+', fl.split('.')[0])))[0] == f[0]]
    print(orig_file)

    list_key_vars = pd.read_csv('list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

    if args.type != 'PPT':
        data = pd.read_csv(f'{orig_folder}/{orig_file[0]}')
        # split data 80/20
        idx = list(set(list(data.index)) - set(index))
        orig_data = data.iloc[idx, :].reset_index(drop=True)
        control_data = data.iloc[index]
        transf_data = pd.read_csv(f'{args.input_folder}/{file}')
    else:
        orig_data = pd.read_csv(f'{orig_folder}/PPT_ARX_train_orig/{orig_file[0]}')
        control_data = pd.read_csv(f'{orig_folder}/PPT_ARX_test_orig/{orig_file[0]}')
        transf_data = pd.read_csv(f'{args.input_folder}/{file}')
        try:
            # transform '*' in np.nan because of data types
            if transf_data[keys[0]].iloc[-1] == '*':
                transf_data = transf_data.replace('*', np.nan)
        except: pass
        transf_data = orig_data.astype(dtype = orig_data.dtypes)

    if args.type=='resampling_and_gans':
        for i in range(len(set_key_vars)):
            try:
                evaluator = LinkabilityEvaluator(ori=orig_data, 
                                            syn=transf_data, 
                                            control=control_data,
                                            n_attacks=len(control_data),
                                            aux_cols=set_key_vars[i],
                                            n_neighbors=10)

                evaluator.evaluate(n_jobs=-2)  # n_jobs follow joblib convention. -1 = all cores, -2 = all execept one
                # evaluator.risk()
                risk = pd.DataFrame({'value': evaluator.risk()[0], 'ci':[evaluator.risk()[1]]})
                risk.to_csv(
                        f'{args.output_folder}/{file.split(".csv")[0]}_qi{i}.csv',
                        index=False)
            except: pass
    else:
        keys_nr = list(map(int, re.findall(r'\d+', file.split('_')[2])))[0]
        print(keys_nr)
        keys = set_key_vars[keys_nr]
        try:
            evaluator = LinkabilityEvaluator(ori=orig_data, 
                                        syn=transf_data, 
                                        control=control_data,
                                        n_attacks=len(control_data),
                                        aux_cols=keys,
                                        n_neighbors=10)

            evaluator.evaluate(n_jobs=-2)  # n_jobs follow joblib convention. -1 = all cores, -2 = all execept one
            # evaluator.risk()
            risk = pd.DataFrame({'value': evaluator.risk()[0], 'ci':[evaluator.risk()[1]]})
            risk.to_csv(
                    f'{args.output_folder}/{file}',
                    index=False)
        except: pass
