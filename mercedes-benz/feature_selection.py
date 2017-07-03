'''some feature selection methods
'''
from sklearn.feature_selection import *
from sklearn.decomposition import PCA
import sklearn
import pandas as pd


def feature_selection(X, y, method="SelectKBest", *args, **kwargs):
    '''ensembled selection function
    '''
    if method == "SelectKBest":
        return SelectKBest(chi2, args, kwargs).fit_transform(X, y)
    elif method == "VarianceThreshold":
        sel = VarianceThreshold(args, kwargs)
        return sel.fit_transform(X)
    elif method == "SelectKFromModel":
        model = SelectFromModel(args, kwargs)
        return model.transform(X)
    elif method = "PCA":
        pca = PCA(args, kwargs)
        return pca.fit_transform(X, y)
    else:
        print("not implemented for now!")
