import os

import pandas as pd
import numpy as np

def get_training_data():
    '''
    Load training data.
    '''
    mainpath = os.environ['MAINPATH']
    datapath = mainpath + '\\data'
    data = pd.read_csv(f'{datapath}\\train.csv')
    data.columns = [col.lower() for col in data.columns]
    
    return data

class FeatureNotFoundError(Exception):
    def __init__(self, feature_name:str):
        message = f'{feature_name} not found in DataFrame.'
        super().__init__(message)

class NumericFeatureMismatchError(Exception):
    def __init__(self, feature_name:str):
        message = f'{feature_name} is not a numeric feature.'
        super().__init__(message)

class TransformerNotFittedError(Exception):
    def __init__(self):
        message = 'Cannot transform before fit method is called.'
        super().__init__(message) 

class TransformerAlreadyFittedError(Exception):
    def __init__(self):
        message = 'Transformer already fitted to data, use overwrite argument to re-fit the pipeline.'
        super().__init__(message) 

class ColumnTransformerTypeMismatchError(Exception):
    def __init__(self, indices:np.ndarray):
        message = f'Arguments at positions {indices} are not FeatureTransformers'
        super().__init__(message)