from typing import List
import pandas as pd
from scipy import cluster
import matplotlib.pyplot as plt
from kangaroo.feature_exploration.feature_exploration import compute_chi2_matrix


def independent_categorical_features(pd_data: pd.DataFrame,
                                     n_features: int, plot_dendrogram: bool = False) -> List[str]:
    """Returns a list of features as different as possible.
    Given a dataframe with categorical columns, returns a list of n_features column names are different as possible.
    The selection algorithm works in two steps:
        - Computes a distance for each pair of features: the result of a chi squared test on both variables.
        - Runs clustering on the computed matrix.

    Args:
        pd_data (pd.DataFrame): dataframe with categorical features
        n_features (int): number of features to select
        plot_dendrogram (bool): if True, plots a dendrogram of all features
    """
    chi2_matrix = compute_chi2_matrix(pd_data)
    z = cluster.hierarchy.ward(chi2_matrix)
    if plot_dendrogram:
        plt.figure(figsize=(20, 10))
        cluster.hierarchy.dendrogram(z, orientation='left')
        locs, _ = plt.yticks()
        plt.yticks(ticks=locs, labels=pd_data.columns)

    cutree_all = cluster.hierarchy.cut_tree(z, n_clusters=n_features)

    dict_features = {}
    for c, n in zip(cutree_all, pd_data.columns):
        dict_features[c[0]] = dict_features.get(c[0], []) + [n]
    return [x[0] for x in dict_features.values()]
