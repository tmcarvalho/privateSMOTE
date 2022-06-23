"""Record linkage
This script applies comparisons between the select QIs for all single outs.
"""
# %%
from os import walk
import gc
import recordlinkage
import pandas as pd
# %%
def record_linkage(transformed: pd.DataFrame,
    original: pd.DataFrame, columns: list) -> pd.DataFrame:
    """_summary_

    Args:
        transformed (pd.DataFrame): transformed dataframe with singleouts
        original (pd.DataFrame): original dataframe with single outs
        columns (list): list of quasi-identifiers

    Returns:
        pd.Dataframe: comparison results with respective score
    """
    indexer = recordlinkage.Index()
    indexer.full()
    candidates = indexer.index(transformed, original)
    print(len(candidates))
    compare = recordlinkage.Compare(n_jobs=-1)
    for idx, col in enumerate(columns):
        if transformed[col].dtype == 'object':
            original[col] = original[col].astype(str)
            compare.string(col, columns[idx], label=columns[idx], method='levenshtein', threshold=0.7)    
        else:
            compare.numeric(col, columns[idx], label=columns[idx], method='gauss')

    comparisons = compare.compute(candidates, transformed, original)
    potential_matches = comparisons[comparisons.sum(axis=1) > 1].reset_index()
    potential_matches['Score'] = potential_matches.iloc[:, 2:].sum(axis=1)
    potential_matches = potential_matches[potential_matches['Score'] >= \
        0.5*potential_matches['Score'].max()]

    return potential_matches


def threshold_record_linkage(transformed_data, original_data, keys):
    """Apply record linkage and calculate the percentage of re-identification

    Args:
        transformed_data (pd.Dataframe): transformed data
        original_data (pd.Dataframe): original dataframe
        keys (list): list of quasi-identifiers

    Returns:
        tuple(pd.Dataframe, list): dataframe with all potential matches,
        list with percentages of re-identification for 50%, 75% and 100% matches
    """
    potential_matches = record_linkage(transformed_data, original_data, keys)

    # get acceptable score (QIs match at least 50%)
    acceptable_score_50 = potential_matches[(potential_matches['Score'] >= \
        0.5*potential_matches['Score'].max()) & (potential_matches['Score'] < \
        0.7*potential_matches['Score'].max())]
    level_1_acceptable_score = acceptable_score_50.groupby(['level_1'])['level_0'].size()
    per_50 = ((1/level_1_acceptable_score.min()) * 100) / len(transformed_data)

    # get acceptable score (QIs match at least 70%)
    acceptable_score_70 = potential_matches[(potential_matches['Score'] >= \
        0.7*potential_matches['Score'].max()) & (potential_matches['Score'] < \
        0.9*potential_matches['Score'].max())]
    level_1_acceptable_score = acceptable_score_70.groupby(['level_1'])['level_0'].size()
    per_70 = ((1/level_1_acceptable_score.min()) * 100) / len(transformed_data)

    # get acceptable score (QIs match at least 90%)
    acceptable_score_90 = potential_matches[(potential_matches['Score'] >= \
        0.9*potential_matches['Score'].max()) & (potential_matches['Score'] < \
        potential_matches['Score'].max())]
    level_1_acceptable_score = acceptable_score_90.groupby(['level_1'])['level_0'].size()
    per_90 = ((1/level_1_acceptable_score.min()) * 100) / len(transformed_data)

    # get max score (all QIs match)
    max_score = potential_matches[potential_matches['Score'] == len(keys)]
    # find original single outs with an unique match in oversampled data - 100% match
    level_1_max_score = max_score.groupby(['level_1'])['level_0'].size()
    per_100 = (len(level_1_max_score[level_1_max_score == 1]) * 100) / len(transformed_data)

    return potential_matches, [per_50, per_70, per_90, per_100]

