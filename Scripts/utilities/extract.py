import numpy as np
import pandas as pd

def get_feature_matrix(masstime, data, significant=True):
    
    # from "data" dataframe, extract significant features
    # based on mass and rtime in "masstime" (series or array)
    # ignore first 3 rows, first 8 columns and return the 
    # float values in a numpy array
    
    if not significant:
        return data.iloc[3:,8:].to_numpy(dtype=float).T
    inds=[]
    for i in range(3,data.shape[0]):
        mt=data.iloc[i,4] + '@' + data.iloc[i,5]
        if mt in masstime:
            inds.append(i)
    inds=np.unique(np.asarray(inds, dtype=int))
    return data.iloc[inds,8:].to_numpy(dtype=float).T

def get_features(masstime, data, significant=True):

    # same as get_feature_matrix, except keep the same data format
    # as the input dataframe
    
    if not significant:
        return data
    inds=[0,1,2]
    for i in range(3,data.shape[0]):
        mt=data.iloc[i,4] + '@' + data.iloc[i,5]
        if mt in masstime:
            inds.append(i)
    inds=np.unique(np.asarray(inds, dtype=int))
    return data.iloc[inds,:]

