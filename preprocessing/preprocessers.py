#%%
import os
os.chdir(os.environ['MAINPATH'])


#%%
import typing as ty
import datetime as dt
import functools as ftools
import math
import copy
import re

import numpy as np
import pandas as pd
import sklearn
sklearn.set_config(display='diagram')
from sklearn import preprocessing as prep
from sklearn import impute
from sklearn import pipeline
from sklearn import compose

from preprocessing.init import TARGET_COL, NUMERIC_FEATURES
from preprocessing.basic_prep import ColumnTransformBuilder, NumericFeatureTransformer

#%%
def build_lotarea_prep() -> NumericFeatureTransformer:
    transformer = NumericFeatureTransformer(feature_name='lotarea')
    imputer = impute.SimpleImputer(strategy='median')
    transformer.add_step('imputation', imputer)
    scaler = prep.StandardScaler()
    transformer.add_step('scaling', scaler)
    return transformer

def build_lotfrontage_prep() -> NumericFeatureTransformer:
    transformer = NumericFeatureTransformer(feature_name='lotfrontage')
    imputer = impute.SimpleImputer(strategy='mean')
    transformer.add_step('imputation', imputer)
    scaler = prep.StandardScaler()
    transformer.add_step('scaling', scaler)
    return transformer

def build_main_transformer() -> compose.ColumnTransformer:
    lotarea_prep = build_lotarea_prep()
    lotfrontage_prep= build_lotfrontage_prep()

    main_transformer = ColumnTransformBuilder(lotarea_prep, lotfrontage_prep).build()
    return main_transformer
