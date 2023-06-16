import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def complete_pairplot(X, label_col=None, feature_tag=None):
    if isinstance(X, pd.DataFrame):
        sns.pairplot(X, vars=X.columns[:-1],hue=label_col)
    else:
        assert(isinstance(X, np.ndarray)
        df_X = pd.DataFrame(X)
        if(feature_tag is not None):
            df_X.columns = feature_tag
        if(isinstance(label_col, np.array)):
            df_X['group']=label_col
            sns.pairplot(df_X, vars=df_X.columns[:-1],hue='group')
        else:
            sns.pairplot(df_X)


        