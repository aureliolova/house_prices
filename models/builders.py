import os
os.chdir(os.environ['MAINPATH'])

import itertools
import contextlib as cm

from sklearn import compose
from sklearn import ensemble
from sklearn import pipeline
from sklearn import linear_model as lm

from models.utils import ModelNotBuiltError, ModelTemplateNotFoundError

class ModelPipelineBuilder:

    def __init__(self, prep_trans: compose.ColumnTransformer, model_template=None, model_kwargs:dict=dict()):
        self.prep_trans = prep_trans
        self.model = None 
        self.model_template = model_template
        self.model_kwargs = model_kwargs 
        self._is_built = False 
    
    def set_model(self, model):
        self.model = model
        self._is_built = True
        self.model_template = 'other'

    def set_model_template(self, model_template):
        self.model_template = model_template

    def set_model_kwarg(self, key:str, val) -> None:
        self.model_kwargs.update({key:val})
    
    def get_model_kwargs(self) -> dict:
        return self.model_kwargs
    
    def build_model(self):
        if not self.model_template:
            raise ModelTemplateNotFoundError()
        built_model = self.model_template(**self.model_kwargs)
        self.model = built_model
        self._is_built = True
        return built_model
    
    def build_pipeline(self) -> pipeline.Pipeline:
        if not self._is_built:
            self.build_model()
        
        pipe = pipeline.Pipeline(
            steps = [
                ('preprocessing', self.prep_trans),
                ('model', self.model)
            ]
        )
        return pipe
    
    def reset(self):
        self.__init__(prep_trans=self.prep_trans)


class ModelTrainingTracker:

    def __init__(self, pipe:pipeline.Pipeline=None):
        self.pipe = pipe
        
    def set_pipe(self, pipe:pipeline.Pipeline):
        self.pipe = pipe
    
    def process_pipe_prep(self):
        '''
        Extract relevant arguments for preprocssing step in model pipeline.
        '''
        prep_column_transformer = dict(self.pipe.steps)['preprocessing']
        transformations = prep_column_transformer.transformers
        trans_steps = [pipe.steps for _, pipe, _ in transformations]
        trans_steps = itertools.chain.from_iterable(trans_steps)
        prep_kwargs = {name:repr(proc) for name, proc in trans_steps}
        self.prep_kwargs = prep_kwargs

    @property
    def pipe(self):
        return self.pipe
    
    @pipe.setter
    def pipe(self, other):
        self.pipe = other
        if self.pipe is None:
            self.prep_kwargs = {}
        else:
            self.process_pipe_prep()

    @pipe.deleter
    def pipe(self):
        self.prep_kwargs = {}
