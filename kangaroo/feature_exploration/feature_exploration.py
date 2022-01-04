import numpy as np
import pandas as pd
from scipy.stats import distributions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


def compute_chi2_matrix(pd_data: pd.DataFrame) -> np.array:
    """Computes a matrix with chi2 tests for each pair of variables.
    When the score for a pair is 0, it means they are highly related. Where it is 1, they are not related.
    :param pd_data: dataframe with N categorical columns
    :return: numpy matrix of size N x N
    """
    dict_unique_values = {}
    for c in pd_data.columns:
        le = LabelEncoder()
        le.fit(pd_data[c])
        pd_data[c] = le.transform(pd_data[c])
        dict_unique_values[c] = len(le.classes_)

    result_tests = np.zeros(shape=(len(pd_data.columns), len(pd_data.columns)))
    for i, c in enumerate(pd_data.columns):
        fs = SelectKBest(score_func=chi2, k='all')
        _ = fs.fit_transform(pd_data.values, pd_data[c])
        degrees_of_freedom = [(dict_unique_values[c] - 1) * (dict_unique_values[x] - 1) for x in pd_data.columns]
        result_tests[i, :] = [distributions.chi2.sf(s, d) for d, s in zip(degrees_of_freedom, fs.scores_)]
    return result_tests
