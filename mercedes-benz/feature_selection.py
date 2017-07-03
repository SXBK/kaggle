'''some feature selection methods
'''
from sklearn.feature_selection import *
from sklearn.decomposition import PCA
import sklearn
import pandas as pd


def feature_selection(X, y, *args, **kwargs):
    '''ensembled selection function
    '''
    if kwargs["method"] is None:
         method="SelectKBest"
    else:
        method = kwargs["method"]
    
    if method.upper() == "u":
        model =  SelectKBest(*args, **kwargs)
        model.fit(X, y)
        return model
    elif method == "VARIANCETHRESHOLD":
        model = VarianceThreshold(*args, **kwargs)
        model.fit(X, y)
        return model
    elif method == "SELECTKFROMMODEL":
        model = SelectFromModel(*args, **kwargs)
        model.fit(X, y)
        return model
    elif method == "PCA":
        model = PCA(*args, **kwargs)
        model.fit(X, y)
        return model
    else:
        print("not implemented for now!")
