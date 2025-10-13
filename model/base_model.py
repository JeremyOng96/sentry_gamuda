from abc import ABC, abstractmethod
from schema.base_model import (
    InferInputTensor,
    InferOutputTensor,
)
from typing import List

class BaseModel(ABC):
    def __init__(self, model_path: str, model_config_path: str):
        self.model_path = model_path
        self.model_config_path = model_config_path
        self.load_model()

    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def inference(self, json_input: dict) -> list[InferOutputTensor]:
        pass

    @abstractmethod
    def json_output(self, model_response: List[InferOutputTensor]) -> dict:
        pass