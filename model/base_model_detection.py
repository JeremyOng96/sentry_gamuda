from abc import ABC, abstractmethod
from schema.base_model import (
    InferInputTensor,
    InferOutputTensor,
)
from typing import List
from model.base_model import BaseModel

class BaseModelDetection(BaseModel):
    def __init__(self, model_path: str, model_config_path: str):
        super().__init__(model_path, model_config_path)
