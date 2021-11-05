import os
import pandas as pd

def get_training_data():
    '''
    Load training data.
    '''
    mainpath = os.environ['MAINPATH']
    datapath = mainpath + '\\data'
    data = pd.read_csv(f'{datapath}\\train.csv')
    data.columns = [col.lower() for col in data.columns]
    
    return data

def get_testing_data():
    '''
    Load training data.
    '''
    mainpath = os.environ['MAINPATH']
    datapath = mainpath + '\\data'
    data = pd.read_csv(f'{datapath}\\test.csv')
    data.columns = [col.lower() for col in data.columns]
    
    return data
    
class ModelNotBuiltError(Exception):
    def __init__(self):
        message = 'the model in this class has not yet been built. Use the build method before building the pipeline.'
        super().__init__(message)

class ModelTemplateNotFoundError(Exception):
    def __init__(self):
        message = 'model template not found, use the set_model_template method.'
        super().__init__(message)