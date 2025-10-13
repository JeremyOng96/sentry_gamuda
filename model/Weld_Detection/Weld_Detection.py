import ultralytics
import cv2
import numpy as np
from schema.base_model import (
    InferInputTensor,
    InferOutputTensor,
    DetectionInstance,
    InferenceResponseJSON
)
from typing import List
from utils.image_utils import decode_image, encode_image
from model.base_model_detection import BaseModelDetection
import logging
import yaml

logger = logging.getLogger(__name__)

class Weld_Detection(BaseModelDetection):
    def __init__(self, model_path: str, yaml_path: str, threshold: float = 0.5):
        super().__init__(model_path, yaml_path)
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            self.class_names = data['names']
            self.class_colors = data['class_colors']
        self.threshold = threshold

    def load_model(self):
        logger.info(f"Loaded Weld Detection model from {self.model_path}")
        self.model = ultralytics.YOLO(self.model_path)

    def inference(self, image: np.ndarray) -> list[InferOutputTensor]:
        """
        Detects objects in an image and returns a list of InferOutputTensor.
        """
        # check if input is empty
        if not image:
            raise ValueError("No image(s) provided")

        results = self.model(image)
        outputs = {
            "output_image": None,
            "output_confidences": [],
            "output_bboxes": [],
            "output_masks": [],
            "output_object_ids": [],
            "output_class_names": [],
        }
        # loop through results
        for result in results:
            img_result = result.plot(conf=True, masks=True)
            boxes = result.boxes
            if boxes is not None:
                # Convert numpy arrays to Python lists and ensure JSON serializable types
                outputs["output_bboxes"].extend(boxes.xyxy.cpu().numpy().tolist())  # [x1, y1, x2, y2]
                outputs["output_confidences"].extend(boxes.conf.cpu().numpy().tolist())
                outputs["output_object_ids"].extend(boxes.cls.cpu().numpy().astype(int).tolist())
                outputs["output_class_names"].extend([self.class_names[cls] for cls in boxes.cls.cpu().numpy().astype(int).tolist()])

            if result.masks is not None:
                # Convert masks to numpy array and then to list
                masks_data = result.masks.data.cpu().numpy().tolist()
                outputs["output_masks"].extend(masks_data)

        # Convert BGR to RGB (result.plot() returns BGR format)
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        image_b64 = encode_image(img_result)
        outputs["output_image"] = image
        output_list = []
        for k, v in outputs.items():
            output_list.append(
                InferOutputTensor(
                    name=k,
                    datatype="FP32",
                    data=v,
                )
            )

        return output_list

    def json_output(self, model_response: List[InferOutputTensor]) -> dict:
        # check if output is empty
        if not model_response:
            raise ValueError("No output data provided")
        
        # Extract data from outputs by name
        output_data = {}
        for output in model_response:
            output_data[output.name] = output.data

        # Get the processed image
        image = output_data.get("output_image", "")

        # Get detection data
        bboxes = output_data.get("output_bboxes", [])
        confidences = output_data.get("output_confidences", [])
        class_ids = output_data.get("output_object_ids", [])
        class_names = output_data.get("output_class_names", [])

        # Build instances array
        instances = []
        compliant = True
        
        for i in range(len(confidences)):
            if i < len(bboxes) and i < len(class_ids):
                bbox = bboxes[i]
                instance = DetectionInstance(
                    x0 = float(bbox[0]),
                    y0 = float(bbox[1]),
                    x1 = float(bbox[2]),
                    y1 = float(bbox[3]),
                    confidence = float(confidences[i]),
                    class_id = int(class_ids[i]),
                    class_name = class_names[i],
                )
                if instance.confidence < self.threshold:
                    compliant = False
                instances.append(instance)

        return InferenceResponseJSON(
            model_name = "Weld_Detection",
            image = image,
            outputs = instances,
        )


