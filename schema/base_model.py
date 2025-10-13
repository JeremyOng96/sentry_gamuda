from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union

# Detection Instance class
class DetectionInstance(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    masks: Optional[Dict[str, Any]] = None  # Changed to match RLE format
    class_id: int
    class_name: str
    image_index: Optional[int] = 0  # Index of the image this detection belongs to

# Segmentation Instance class
class SegmentationInstance(BaseModel):
    masks: Dict[str, Any]
    class_id: int
    class_name: str

# OCR Instance class
class OCRInstance(BaseModel):
    text: str
    score_text: Optional[float] = None
    score_link: Optional[float] = None

# Open Inferece Protocol specific classes
class InferInputTensor(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    data: Any
    parameters: Optional[Dict[str, Any]] = None

class InferOutputTensor(BaseModel):
    name: str
    shape: Optional[List[int]] = None
    datatype: Optional[str] = None
    data: Any
    parameters: Optional[Dict[str, Any]] = None

class InferenceRequestJSON(BaseModel):
    id: str
    parameters: Optional[Dict[str, Any]] = None
    inputs: List[InferInputTensor]
    outputs: Optional[List[InferOutputTensor]] = None # Optional specifications of the required output tensors. If not specified all outputs produced by the model will be returned
    
class InferenceResponseJSON(BaseModel):
    id: Optional[str] = None
    model_name: str
    model_version: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    image: Any = None
    outputs: List[Union[DetectionInstance,SegmentationInstance,OCRInstance]] = []
