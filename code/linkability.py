#!/usr/bin/env python
import pandas as pd
import argparse
from anonymeter.evaluators import LinkabilityEvaluator


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--orig_file', type=str, default="none")
parser.add_argument('--transf_file', type=str, default="none")
parser.add_argument('--control_file', type=str, default="none")
parser.add_argument('--key_vars', type=str, default="none")
args = parser.parse_args()


data = pd.read_csv(f'{args.orig_file}')

transf_data = pd.read_csv(f'synth_data/{args.transf_file}')

control_data = pd.read_csv(f'{args.control_file}')

evaluator = LinkabilityEvaluator(ori=data,
                            syn=transf_data,
                            control=control_data,
                            n_attacks=len(control_data),
                            aux_cols=args.keys,
                            n_neighbors=10)

evaluator.evaluate(n_jobs=-1)
risk = pd.DataFrame({'value': evaluator.risk()[0], 'ci':[evaluator.risk()[1]]})
risk.to_csv(
        f'linkabilty_results/{args.transf_file}',
        index=False)
