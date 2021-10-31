#%% env setup
import os
os.chdir(os.environ['MAINPATH'])

#%% imports
import datetime as dt
import math
import copy
import re
import typing as ty

import pandas as pd
import numpy as np

from sklearn import base
from sklearn import compose
from sklearn import pipeline
from sklearn import preprocessing as prep

from preprocessing.utils import TransformerAlreadyFittedError, TransformerNotFittedError, FeatureNotFoundError, NumericFeatureMismatchError
#%%
class NumericFeatureTransformer(base.TransformerMixin, base.BaseEstimator):
    '''
    Base class for transforming a specific numeric feature.
    '''
    def __init__(self, feature_name:str=None):
        self.feature_name = feature_name
        self._step_names = []
        self._step_trans = []
        self._is_fitted = False

    def __add__(self, other) -> pipeline.Pipeline:
        return self.__radd__(other)


    def __radd__(self, other) -> pipeline.Pipeline:
        s_trans = self.make_pipeline()
        s_feat = self.feature_name

        o_trans = other.make_pipeline()
        o_feat = other.feature_name

        sum_pipe_steps = [
            (s_feat, s_trans),
            (o_feat, o_trans),
        ]
        sum_pipe = pipeline.Pipeline(sum_pipe_steps)
        return sum_pipe 

    def __ladd__(self, other) -> pipeline.Pipeline:
        s_trans = self.make_pipeline()
        s_feat = self.feature_name

        o_trans = other.make_pipeline()
        o_feat = other.feature_name

        sum_pipe_steps = [
            (o_feat, o_trans),
            (s_feat, s_trans)
        ]
        sum_pipe = pipeline.Pipeline(sum_pipe_steps)
        return sum_pipe  


    def set_feature_name(self, feature_name:str) -> None:
        '''
        Set the feature name to transform.
        '''
        self.feature_name = feature_name

    def add_step(self, step_name:str, step_trans:ty.Union[base.TransformerMixin, base.BaseEstimator]):
        '''
        Add a step to the transformation pipeline
        '''
        
        self._step_names.append(f'{self.feature_name}_{step_name}')
        self._step_trans.append(step_trans)
        

    def fit(self, data:pd.DataFrame, overwrite:bool=False):
        '''
        Fit the current pipeline to data.
        '''
        self._feature_check(data)
        if self._is_fitted and not overwrite:
            raise TransformerAlreadyFittedError()
        self._pipe = self.make_pipeline()
        self._pipe.fit(data)
        self._is_fitted = True
        return self

    def transform(self, data:pd.DataFrame):
        '''
        Transform data after fitting the current pipeline
        '''
        if not self._is_fitted:
            raise TransformerNotFittedError()
        trans_data = self._pipe.transform(data)

        return trans_data
    
    def fit_transform(self, data:pd.DataFrame, overwrite:bool=False):
        ''' 
        Apply fit and transform methods consecutively.
        '''
        self.fit(data, overwrite=overwrite)
        trans_data = self.transform(data)
        return trans_data

    
    def make_pipeline(self):
        if not self._step_names:
            self._handle_empty_pipeline()
        
        pipe_steps = self.make_pipeline_steps()
        pipe = pipeline.Pipeline(steps=pipe_steps)
        trans_tuple = (f'{self.feature_name}_trans', pipe, [self.feature_name])
        transformer = compose.ColumnTransformer(
            [trans_tuple]
        )

        return transformer    

    def make_pipeline_steps(self):
        pipe_steps = list(zip(self._step_names, self._step_trans))
        return pipe_steps

    def _handle_empty_pipeline(self):
        identity_trans = prep.FunctionTransformer()
        self.add_step(step_name='identity', step_trans= identity_trans)

    def _feature_check(self, data:pd.DataFrame):
        self._feature_existence_check(data)
        self._feature_type_check(data)
        
    def _feature_existence_check(self, data:pd.DataFrame):
        if self.feature_name not in data.columns:
            raise FeatureNotFoundError(self.feature_name)
    
    def _feature_type_check(self, data:pd.DataFrame):
        feature_dtype = data.loc[:, self.feature_name].dtype
        if not np.issubdtype(feature_dtype, np.number):
            raise NumericFeatureMismatchError(self.feature_name)

# %%
