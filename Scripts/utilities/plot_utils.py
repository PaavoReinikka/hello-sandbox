import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def complete_pairplot(X, group_col=None, feature_ids=None, kind='kde', palette=None):
    if isinstance(X, pd.DataFrame):
        if(group_col is None):
            group_col='group'
        
        sns.pairplot(data=X,
            vars=X.columns[:-1],
            kind='scatter',
            hue=group_col,
            palette=palette,
            diag_kind=kind)
                
    else:
        df_X = pd.DataFrame(X)
        if(feature_ids is None):
            feature_ids = ["V{}".format(i) for i in range(X.shape[1])]

        df_X.columns = feature_ids
        df_X['group']=group_col
        
        sns.pairplot(data=df_X,
            vars=df_X.columns[:-1],
            kind='scatter',
            hue='group',
            palette=palette,
            diag_kind=kind)

    