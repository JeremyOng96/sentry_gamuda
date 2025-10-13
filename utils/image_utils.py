import cv2
import base64
import numpy as np

def decode_image(image_data: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)

def encode_image(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8') 

def hex_to_rgb(hex_color):
    """Convert hexadecimal color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def draw_boxes(image, boxes, confidences, class_names, colors):
    """Draw bounding boxes and labels on the image"""
    img = image.copy()
    for box, conf, name, color in zip(boxes, confidences, class_names, colors):
        try:
            # COCO format: [x, y, width, height]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[0] + box[2])
            y2 = int(box[1] + box[3])
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid box coordinates: {box}, error: {e}")
            continue
        
        if isinstance(color, str):
            color = hex_to_rgb(color)
        
        if not isinstance(color, tuple) or len(color) not in (3, 4) or not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            logger.error(f"Invalid color format: {color}")
            continue
        
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        try:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_height - baseline - 5), (x1 + label_width, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        except cv2.error as e:
            logger.error(f"OpenCV error in draw_boxes: {e}")
            continue
    return img