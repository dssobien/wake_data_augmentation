#!python3

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def df_L2_norm(df, cols):
    # get only the values of interest
    values = df[cols].values
    # return the L2 norm for the values
    return L2_norm(values)


def L2_norm(values):
    # square the values
    values *= values
    # sum the rows of the data
    values = values.sum(-1)
    # return the square root of each value
    return np.sqrt(values)


def df_distance(df, cols, metric='euclidean'):
    # ensure dataframe is just columns of interest
    df = df[cols]
    names = df.index
    # get array of values from the dataframe
    v = df.values

    df_d = pd.DataFrame(cdist(v, v, metric), columns=names, index=names).replace(0, np.nan)    
    # we add a column for the closest coordinate
    df_d["min_col"] = df_d.idxmin()
    
    # we want to also know more than just the closest coordinate, so we'll sort
    # distances and used their indices to created a list
    sorted_lists = []
    for i in range(len(names)):
        # grab row i, cut the last column value, and convert to floating point
        v = df_d.iloc[i].values[:-1].astype('float')
        # convert nan values to inf so they sort to last position
        v[np.isnan(v)] = np.inf
        # sort distance values, then use their index to get sorted list of
        # augmentation sets, cut the last one off b/c it is the same set
        sorted_lists.append(list(names[np.argsort(v)][:-1].values))
    df_d["sorted_lists"] = sorted_lists
    return df_d
