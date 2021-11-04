import os
os.chdir(os.environ['MAINPATH'])

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


    
    

