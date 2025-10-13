import gradio as gr
import numpy as np
import cv2
import ultralytics
from ultralytics import YOLO
from typing import List
import logging
import time
import json
from datetime import datetime
from gradio_image_annotation import image_annotator
import yaml
import matplotlib.pyplot as plt
import matplotlib
import ast
matplotlib.use('Agg')  # Use non-interactive backend

from model.Weld_Detection.Weld_Detection import Weld_Detection
from report.report_generation import generate_report_from_predictions
from database.db_utils import get_projects_from_database, get_project_details, get_users_from_database
from utils.image_utils import draw_boxes, hex_to_rgb


css = """
footer { display: none !important; }
.gradio-info { display: none !important; }
"""


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize model once since it's read-only and can be shared
model_class = Weld_Detection("./model/Weld_Detection/best.pt", "./model/Weld_Detection/weld_detection.yaml")
model_class.load_model()

# These are read-only configuration values that can be shared
label_list = list(model_class.class_names.values())
label_colors = [(ast.literal_eval(color)) for color in model_class.class_colors.values()]
confidence_threshold = 0.6

def get_initial_predictions():
    """Get initial empty predictions state for each user session"""
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": label, "supercategory": "weld"} 
            for i, label in enumerate(label_list)
        ]
    }

def process_image(file):
    """Process a single image file into numpy array format"""
    try:
        with open(file.name, 'rb') as f:
            img_bytes = f.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            logger.error(f"Failed to decode image: {file.name}")
            return None
        return img_array
    except Exception as e:
        logger.error(f"Failed to load {file.name}: {e}")
        return None

def model_inference(files, predictions_state):
    """Process multiple image files and run model inference"""    
    
    # Reset predictions for this session
    predictions_state = get_initial_predictions()
    
    low_confidence_annotations = []
    
    if not files:
        logger.warning("No files uploaded")
        return None, low_confidence_annotations
    
    start_time = time.time()
    logger.info(f"Processing {len(files)} images")
    
    annotated_images = []

    for file_idx, file in enumerate(files):
        img_array = process_image(file)
        if img_array is None:
            continue
            
        # Add image info to COCO format
        image_info = {
            "id": file_idx + 1,
            "file_name": file.name.split('/')[-1],
            "width": img_array.shape[1],
            "height": img_array.shape[0],
            "date_captured": datetime.now().strftime("%Y-%m-%d"),
        }
        predictions_state["images"].append(image_info)

        infer_output = model_class.inference([img_array])
        json_results = model_class.json_output(infer_output)
        
        # Initialize annotated image
        annotated_image = img_array.copy()
        
        # Process each detection
        for index, instance in enumerate(json_results.outputs):
            if not (0 <= instance.class_id < len(label_colors)):
                logger.error(f"Invalid class_id: {instance.class_id}")
                continue

            # Create annotation in COCO format
            prediction = {
                "id": len(predictions_state["annotations"]) + 1,  # Unique ID for each annotation
                "image_id": file_idx + 1,
                "category_id": instance.class_id,
                "bbox": [
                    float(instance.x0),
                    float(instance.y0),
                    float(instance.x1 - instance.x0),
                    float(instance.y1 - instance.y0)
                ],
                "area": float(instance.x1 - instance.x0) * float(instance.y1 - instance.y0),
                "iscrowd": 0,
                "segmentation": [],
                "confidence": float(instance.confidence)
            }
            predictions_state["annotations"].append(prediction)

            color = label_colors[instance.class_id]
            if isinstance(color, str):
                color = hex_to_rgb(color)
            
            annotated_image = draw_boxes(
                annotated_image,
                [prediction["bbox"]],
                [prediction["confidence"]],
                [label_list[instance.class_id]],
                [color]
            )
            
            if prediction["confidence"] < confidence_threshold:
                low_confidence_annotations.append(image_info["file_name"])

        if annotated_image is not None:
            annotated_images.append(annotated_image)
    
    low_confidence_annotations = list(set(low_confidence_annotations))
    
    end_time = time.time()
    logger.info(f"Model inference completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Number of annotated images: {len(annotated_images)}")
    logger.info(f"Number of low confidence annotations: {len(low_confidence_annotations)}")
    logger.info(f"Low confidence annotations: {low_confidence_annotations}")
    return annotated_images if annotated_images else None, low_confidence_annotations, predictions_state

def update_predictions(annotator_output, low_conf_filenames, selected_index, files, output_images, predictions_state):
    """Update predictions with edited annotations and redraw the gallery"""

    logger.info(f"low_conf_filenames: {len(low_conf_filenames) if low_conf_filenames else 0}")
    
    try:
        if selected_index:
            image_info = next((img for img in predictions_state["images"] if img["file_name"] == selected_index), None)
            if not image_info:
                logger.error(f"Could not find image info for: {selected_index}")
                return output_images, low_conf_filenames, predictions_state

            edited_boxes = annotator_output.get("boxes", [])
            logger.info(f"Edited boxes: {len(edited_boxes)}")
            
            # Remove existing annotations for this image
            predictions_state["annotations"] = [
                ann for ann in predictions_state["annotations"]
                if ann["image_id"] != image_info["id"]
            ]
            
            # Add new annotations
            for box in edited_boxes:
                category_id = next((cat["id"] for cat in predictions_state["categories"] if cat["name"] == box["label"]))
                
                # Create new annotation
                annotation = {
                    "id": len(predictions_state["annotations"]) + 1,
                    "image_id": image_info["id"],
                    "category_id": category_id,
                    "bbox": [
                        float(box["xmin"]), # top left corner
                        float(box["ymin"]), # top left corner
                        float(box["xmax"] - box["xmin"]), # width
                        float(box["ymax"] - box["ymin"]) # height
                    ],
                    "area": float((box["xmax"] - box["xmin"]) * (box["ymax"] - box["ymin"])),
                    "iscrowd": 0,
                    "segmentation": [],
                    "confidence": 1.0  
                }
                predictions_state["annotations"].append(annotation)
            
            file_path = next(f.name for f in files if f.name.split('/')[-1] == selected_index)
            img_array = process_image(next(f for f in files if f.name == file_path))
            
            if img_array is not None:
                image_annotations = [
                    ann for ann in predictions_state["annotations"] 
                    if ann["image_id"] == image_info["id"]
                ]
                
                # Draw boxes
                annotated_image = img_array.copy()
                for ann in image_annotations:
                    category = next(cat for cat in predictions_state["categories"] if cat["id"] == ann["category_id"])
                    annotated_image = draw_boxes(
                        annotated_image,
                        [ann["bbox"]],
                        [ann.get("confidence", 1.0)],
                        [category["name"]],
                        [label_colors[ann["category_id"]]]
                    )
                
                file_index = next(i for i, f in enumerate(files) if f.name == file_path)
                if isinstance(output_images, list) and 0 <= file_index < len(output_images):
                    output_images[file_index] = annotated_image
            
            logger.info(f"Updated predictions for image {selected_index}")
            return output_images, low_conf_filenames, predictions_state
    
    except Exception as e:
        logger.error(f"Error updating predictions: {str(e)}")
    
    return output_images, low_conf_filenames, predictions_state

def update_dropdown(low_conf_annotations, files):
    """Update dropdown with available low confidence images"""
    num_items = len(files)
    logger.info(f"low_conf_annotations: {low_conf_annotations}")
    temp_files = [file.name.split('/')[-1] for file in files]
    choices = list(set(low_conf_annotations) & set(temp_files))

    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

def update_annotator(low_conf_filenames, selected_index, files, path_map, predictions_state):
    """Update the image annotator with the selected low confidence image"""
    logger.info(f"Updating image_annotator with selected_index: {selected_index}")
    
    # Validate inputs
    if not all([low_conf_filenames, selected_index, files, path_map, predictions_state]):
        logger.warning("Missing required inputs for update_annotator")
        return None
        
    try:
        # Handle selected_index being a list or single value
        if isinstance(selected_index, list):
            selected_index = selected_index[0] if selected_index else None
            
        if not selected_index:
            logger.warning("No valid selected_index")
            return None
            
        # Get the full path from the path map
        full_path = path_map.get(selected_index)
        if not full_path:
            logger.error(f"Could not find full path for filename: {selected_index}")
            return None
            
        # Find image info and annotations
        image_info = next((img for img in predictions_state["images"] if img["file_name"] == selected_index), None)
        if not image_info:
            logger.error(f"Could not find image info for: {selected_index}")
            return None
            
        # Get all annotations for this image
        image_annotations = [
            ann for ann in predictions_state["annotations"] 
            if ann["image_id"] == image_info["id"]
        ]
        
        # Get category mapping
        category_map = {cat["id"]: cat["name"] for cat in predictions_state["categories"]}
        
        # Load image
        img_array = process_image(next(f for f in files if f.name == full_path))
        if img_array is None:
            return None
            
        formatted_prediction = {
            "image": img_array,
            "boxes": [
                {
                    "xmin": int(float(ann["bbox"][0])),
                    "ymin": int(float(ann["bbox"][1])),
                    "xmax": int(float(ann["bbox"][0] + ann["bbox"][2])),
                    "ymax": int(float(ann["bbox"][1] + ann["bbox"][3])),
                    "label": category_map[ann["category_id"]],
                    "color": label_colors[ann["category_id"]] if isinstance(label_colors[ann["category_id"]], tuple) else hex_to_rgb(label_colors[ann["category_id"]]),
                }
                for ann in image_annotations
            ]
        }
        
        logger.info(f"Successfully formatted prediction for image_annotator")
        return formatted_prediction
        
    except Exception as e:
        logger.error(f"Error in update_annotator: {str(e)}")
        return None

def inference(files, predictions_state):
    """Process images and update dropdown choices"""
    
    # Reset predictions state when new files are uploaded
    predictions_state = get_initial_predictions()
    
    if not files:
        logger.info("No files provided for inference")
        return None, [], {}, gr.Dropdown(choices=[], value=None), predictions_state
        
    annotated_images, low_conf_annotations, predictions_state = model_inference(files, predictions_state)
    logger.info(f"Returning from model_inference: {len(low_conf_annotations)} low-confidence annotations")
    
    # Update dropdown choices and path mapping
    choices = []
    path_map = {}
    
    if low_conf_annotations:
        # Store file indices and names for low confidence predictions
        for i, f in enumerate(files[:len(low_conf_annotations)]):
            if hasattr(f, 'name'):
                # Extract just the filename from the path
                filename = f.name.split('/')[-1]
                full_path = f.name
                choices.append(filename)
                path_map[filename] = full_path
                logger.info(f"Added file to choices: {filename} -> {full_path}")
        
        logger.info(f"Generated dropdown choices: {choices}")
        logger.info(f"Generated path map: {path_map}")
        
        return (
            annotated_images,
            low_conf_annotations,
            path_map,
            gr.Dropdown(choices=choices, value=choices[0] if choices else None),
            predictions_state
        )
    else:
        logger.info("No low confidence annotations found")
        return (
            annotated_images,
            low_conf_annotations,
            {},
            gr.Dropdown(choices=[], value=None),
            predictions_state
        )

# Function to display current annotations in real-time
def create_weld_statistics_chart(predictions_state):
    """Create a matplotlib chart showing weld detection statistics"""
    
    if not predictions_state or not predictions_state.get("annotations"):
        # Create empty chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.text(0.5, 0.5, 'No data available\nProcess images first', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Weld Quality Distribution')
        ax2.text(0.5, 0.5, 'No data available\nProcess images first', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Confidence Distribution')
        plt.tight_layout()
        return fig
    
    # Count weld types and confidence levels
    good_weld = 0
    bad_weld = 0
    low_confidence = 0
    high_confidence = 0
    all_confidences = []
    
    # Get category mapping
    category_map = {cat["id"]: cat["name"] for cat in predictions_state["categories"]}
    
    for ann in predictions_state["annotations"]:
        conf = ann.get("confidence", 1.0)
        all_confidences.append(conf)
        
        if conf >= confidence_threshold:
            high_confidence += 1
            category_name = category_map.get(ann["category_id"])
            if category_name == "good_weld":
                good_weld += 1
            elif category_name == "bad_weld":
                bad_weld += 1
        else:
            low_confidence += 1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart for weld quality distribution
    if good_weld + bad_weld > 0:
        labels = ['Good Weld', 'Bad Weld', 'Low Confidence']
        sizes = [good_weld, bad_weld, low_confidence]
        colors = ['#2ecc71', '#e74c3c', '#f1c40f']  # Green for good, red for bad, yellow for low confidence    
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Weld Quality Distribution')
    else:
        ax1.text(0.5, 0.5, 'No high-confidence\ndetections', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Weld Quality Distribution')
    
    # Bar chart for confidence distribution
    if all_confidences:
        ax2.hist(all_confidences, bins=10, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Number of Detections')
        ax2.set_title(f'Confidence Distribution\n(Total: {len(all_confidences)} detections)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No detections\navailable', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Confidence Distribution')
    
    plt.tight_layout()
    return fig

def display_current_annotations(low_conf_filenames, files, predictions_state):
    """Display current annotations in JSON format"""
    
    # Ensure low_conf_filenames is a list
    if not isinstance(low_conf_filenames, (list, tuple)):
        logger.warning(f"low_conf_filenames is not a list: {type(low_conf_filenames)}")
        low_conf_filenames = []
    
    # Check for required data
    if not files or not predictions_state:
        logger.info("Missing required data for displaying annotations")
        return {
            "num_low_confidence_images": 0,
            "annotations": []
        }
    
    try:
        display_data = {
            "num_low_confidence_images": len(low_conf_filenames),
            "annotations": []
        }
        
        # Get annotations for low confidence images
        for filename in low_conf_filenames:
            try:
                # Find image ID for this filename
                image_info = next((img for img in predictions_state["images"] if img["file_name"] == filename), None)
                if not image_info:
                    logger.warning(f"Could not find image info for filename: {filename}")
                    continue
                    
                image_id = image_info["id"]
                
                # Get all annotations for this image
                image_annotations = [
                    ann for ann in predictions_state["annotations"] 
                    if ann["image_id"] == image_id
                ]
                
                formatted_annotation = {
                    "image_name": filename,
                    "image_index": image_id - 1,  # Convert to 0-based index
                    "boxes": []
                }
                
                for ann in image_annotations:
                    try:
                        category_name = next(
                            (cat["name"] for cat in predictions_state["categories"] if cat["id"] == ann["category_id"]),
                            "unknown"
                        )
                        formatted_box = {
                            "coordinates": {
                                "xmin": float(ann["bbox"][0]),
                                "ymin": float(ann["bbox"][1]),
                                "xmax": float(ann["bbox"][0] + ann["bbox"][2]),
                                "ymax": float(ann["bbox"][1] + ann["bbox"][3])
                            },
                            "label": category_name,
                            "confidence": float(ann.get("confidence", 1.0))
                        }
                        formatted_annotation["boxes"].append(formatted_box)
                    except (KeyError, IndexError, ValueError) as e:
                        logger.warning(f"Error formatting annotation for {filename}: {e}")
                        continue
                
                display_data["annotations"].append(formatted_annotation)
                
            except Exception as e:
                logger.warning(f"Error processing file {filename}: {e}")
                continue
        
        return display_data
        
    except Exception as e:
        logger.error(f"Error in display_current_annotations: {e}", exc_info=True)
        return {}

def export_predictions(form_date, description, created_by, status, predictions_state):
    """Export ALL predictions in COCO format"""
    
    if not files:
        return None
    
    # Handle date conversion
    if form_date:
        if isinstance(form_date, (int, float)):
            date_obj = datetime.fromtimestamp(form_date)
            formatted_date = date_obj.strftime("%Y-%m-%d")
        elif hasattr(form_date, 'strftime'):
            formatted_date = form_date.strftime("%Y-%m-%d")
        elif isinstance(form_date, str):
            try:
                date_obj = datetime.strptime(form_date, "%Y-%m-%d %H:%M:%S")
                formatted_date = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                formatted_date = form_date
        else:
            formatted_date = str(form_date)
    else:
        formatted_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create export data in COCO format
    export_data = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": description or "Gamuda Weld Detection Dataset",
            "contributor": created_by or "Gamuda IBS",
            "date_created": formatted_date,
            "url": "https://parexus.ai/"
        },
        "licenses": [
            {
                "url": "https://parexus.ai/",
                "id": 1,
                "name": "Proprietary"
            }
        ],
        "images": predictions_state["images"],
        "annotations": predictions_state["annotations"],
        "categories": predictions_state["categories"]
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_weld_predictions_coco_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return filename

def generate_pdf_report(files, location, form_date, description, created_by, status, due_date=None, qa_description=None, qa_created_by=None, predictions_state=None, project_value=None):
    """Generate PDF report from current predictions and project information"""
    
    logger.info("Starting PDF report generation...")
    logger.info(f"Files: {len(files) if files else 0}, Predictions: {len(predictions_state.get('annotations', [])) if predictions_state else 0}")
    
    if not files or not predictions_state or not predictions_state.get("annotations"):
        logger.error("No files or predictions available for PDF generation")
        return None
    
    # Handle date conversion - Gradio DateTime returns timestamp as float
    if form_date:
        if isinstance(form_date, (int, float)):
            # Convert timestamp to datetime object
            date_obj = datetime.fromtimestamp(form_date)
            formatted_date = date_obj.strftime("%Y-%m-%d")
        elif hasattr(form_date, 'strftime'):
            # Already a datetime object
            formatted_date = form_date.strftime("%Y-%m-%d")
        elif isinstance(form_date, str):
            # Try to parse the string
            try:
                date_obj = datetime.strptime(form_date, "%Y-%m-%d %H:%M:%S")
                formatted_date = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                formatted_date = form_date
        else:
            # Fallback to string representation
            formatted_date = str(form_date)
    else:
        formatted_date = datetime.now().strftime("%Y-%m-%d")
        
    # Handle due_date conversion
    if due_date:
        if isinstance(due_date, (int, float)):
            # Convert timestamp to datetime object
            due_date_obj = datetime.fromtimestamp(due_date)
            formatted_due_date = due_date_obj.strftime("%Y-%m-%d")
        elif hasattr(due_date, 'strftime'):
            # Already a datetime object
            formatted_due_date = due_date.strftime("%Y-%m-%d")
        elif isinstance(due_date, str):
            # Try to parse the string
            try:
                due_date_obj = datetime.strptime(due_date, "%Y-%m-%d %H:%M:%S")
                formatted_due_date = due_date_obj.strftime("%Y-%m-%d")
            except ValueError:
                formatted_due_date = due_date
        else:
            # Fallback to string representation
            formatted_due_date = str(due_date)
    else:
        formatted_due_date = "Not specified"    

    # Get category mapping
    category_map = {cat["id"]: cat["name"] for cat in predictions_state["categories"]}
    
    # Count by category and confidence
    good_welds = 0
    bad_welds = 0
    low_confidence = 0
    edited_welds = 0
    
    for ann in predictions_state["annotations"]:
        conf = ann.get("confidence", 1.0)
        category_name = category_map.get(ann["category_id"])
        
        if conf < confidence_threshold:
            low_confidence += 1
        elif category_name == "good_weld":
            good_welds += 1
        elif category_name == "bad_weld":
            bad_welds += 1
            
        if conf == 1.0:  # Edited annotations
            edited_welds += 1
    
    total_welds = good_welds + bad_welds + low_confidence
    
    # Get project ID from the project value
    logger.info(f"Project value: {project_value}")
    
    if not project_value:
        logger.warning("No project selected. Please select a project before generating the report.")
        return None
    try:
        project_id = int(project_value.split(" - ")[0])
        logger.info(f"Successfully parsed project ID: {project_id}")
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing project ID: {e}")
        return None

    metadata = {
        "project_id": project_id,
        "project_manager_id": created_by.split("-")[0],
        "created_by": created_by.split("-")[1],
        "location": location,
        "date": formatted_date,
        "due_date": formatted_due_date,
        "description": description or "Not specified", 
        "status": status or "Not specified",
        "export_time": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "statistics": {
            "good_welds": good_welds,
            "bad_welds": bad_welds,
            "low_confidence": low_confidence,
            "total_welds": total_welds,
            "edited_welds": edited_welds
        },
        "qa_description": qa_description if qa_description else "Not specified",
        "qa_user_id": qa_created_by.split("-")[0],
        "qa_created_by": qa_created_by.split("-")[1],
        "labels": label_list,
        "colors": label_colors,
    }
                
    # Generate PDF report using the report generation function
    report_path = generate_report_from_predictions(
        all_predictions=predictions_state,
        files=files,
        metadata=metadata
    )
    
    if report_path:
        logger.info(f"PDF report generated successfully: {report_path}")

        # Get project ID and user ID from metadata (we already validated it exists)
        try:
            project_id = metadata.get('project_id')
            qa_user_id = metadata.get('qa_user_id')
            logger.info(f"Using project ID: {project_id}, QA User ID: {qa_user_id}")
            
            if project_id and qa_user_id:
                # Get statistics from metadata
                stats = metadata.get('statistics', {})
                good_welds = stats.get('good_welds', 0)
                bad_welds = stats.get('bad_welds', 0)
                low_confidence = stats.get('low_confidence', 0)
                
                # Push results to database
                from database.db_utils import push_results_to_database
                success = push_results_to_database(
                    project_id=project_id,
                    user_id=qa_user_id,  # Use the numeric user ID
                    description=qa_description if qa_description else description,
                    good_weld=good_welds,
                    bad_weld=bad_welds,
                    uncertain=low_confidence
                )
                
                if success:
                    logger.info("Successfully saved results to database")
                else:
                    logger.error("Failed to save results to database")
        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
    else:
        logger.error("Failed to generate PDF report - report_path is None")
    
    return report_path

with gr.Blocks(css=css, title="Sentry") as demo:
    # Initialize session state
    predictions_state = gr.State(get_initial_predictions())
    
    gr.Image("./assets/gamuda-logo-header-red.png", label="", show_label=False, container=False, height=100, width=300, show_download_button=False, show_fullscreen_button=False)
    gr.Markdown("<h1 style='text-align: left;'>Gamuda Weld Quality Detection</h1>")
    
    with gr.Row():
        project_dropdown = gr.Dropdown(
            label="Select Project",
            choices=get_projects_from_database(),
            interactive=True,
            allow_custom_value=False,
            type="value",
            multiselect=False,
            value=None  # Don't pre-select anything
        )

    low_conf_state = gr.State(value=[])
    file_path_map = gr.State({})

    with gr.Accordion("Project Information", open=False):
        project_location = gr.Textbox(
            label="Project Location", 
            placeholder="e.g., Site A, Building 4", 
            interactive=False,
        )
        project_start_date = gr.DateTime(
            label="Project Start Date",
            interactive=False,
        )
        project_end_date = gr.DateTime(
            label="Project End Date",
            interactive=False,
        )
        project_description = gr.Textbox(
            label="Project Description", 
            lines=3, 
            placeholder="Describe the scope and goals of the project...", 
            interactive=False,
        )
        project_created_by = gr.Textbox(
            label="Project Created By", 
            placeholder="Enter name of the project manager...", 
            interactive=False,
        )
        project_status = gr.Dropdown(
            label="Overall Project Status",
            choices=["Not started", "In Progress", "Done"], 
            interactive=False,
        )

    with gr.Accordion("Quality Assurance Information", open=True):
        qa_description = gr.Textbox(
            label="QA Report Notes / Description", 
            lines=3,
            placeholder="Enter details about this specific inspection...", 
            interactive=True
        )
        qa_created_by = gr.Dropdown(
            label="QA Report Submitted By", 
            choices=get_users_from_database(),
            interactive=True,
            allow_custom_value=False,
            type="value",
            multiselect=False
        )

    with gr.Row():
        with gr.Column(scale=2):
            input_images = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Upload Weld Images",
                height="300px"
            )
            predict_button = gr.Button("Process Images", variant="primary")
            output_images = gr.Gallery(
                label="Detected Welds",
                show_label=True,
                height="auto",
                columns=6,
                rows=2,
                interactive=False,
            )
            statistics_plot = gr.Plot(label="Weld Detection Statistics")

    gr.Markdown("<h2 style='text-align: left;'>Weld Quality Editing</h1>")
    with gr.Row(equal_height=True):
        with gr.Column():
            low_conf_index = gr.Dropdown(
                label="Select Low-Confidence Image",
                choices=[],
                interactive=True,
                allow_custom_value=False,
                type="value",
                multiselect=False
            )

            annotator = image_annotator(
                value=None,
                label_list=label_list,
                label_colors=label_colors,
                label="Annotate Low-Confidence Image",
                show_label=True,
            )

            live_json = gr.JSON(label="Current Annotations (Live)", value={})
            with gr.Column():
                update_button = gr.Button("Update annotations", variant="primary")
                
    # Add a file component for downloads that's visible
    download_file = gr.File(label="Download JSON File", visible=True)
    download_button = gr.DownloadButton("Download All Predictions (JSON)", variant="secondary")
    
    # Add PDF report download
    pdf_download_file = gr.File(label="Download PDF Report", visible=True)
    pdf_download_button = gr.DownloadButton("Generate PDF Report", variant="primary")

    gr.Markdown("""
    - `C`: Create mode (add new bounding box)
    - `D`: Drag mode (move existing box)
    - `E`: Edit selected box (same as double-click a box)
    - `Delete`: Remove selected box
    - `Space`: Reset view (zoom/pan)
    - `Enter`: Confirm modal dialog (e.g., save annotations)
    - `Escape`: Cancel/close modal dialog
    """)
    
    # Connect all events
    predict_button.click(
        fn=inference,
        inputs=[input_images, predictions_state],
        outputs=[output_images, low_conf_state, file_path_map, low_conf_index, predictions_state]
    ).then(
        fn=lambda x, y, z: display_current_annotations(x or [], y, z),  # Show initial annotations
        inputs=[low_conf_state, input_images, predictions_state],
        outputs=[live_json]
    ).then(
        fn=create_weld_statistics_chart,  # Update statistics chart
        inputs=[predictions_state],
        outputs=[statistics_plot]
    )

    low_conf_state.change(
        fn=update_dropdown,
        inputs=[low_conf_state, input_images],
        outputs=[low_conf_index]
    )

    low_conf_index.change(
        fn=update_annotator,
        inputs=[low_conf_state, low_conf_index, input_images, file_path_map, predictions_state],
        outputs=[annotator]
    )

    # Update button: Apply changes and refresh both gallery and JSON display
    update_button.click(
        fn=update_predictions,
        inputs=[annotator, low_conf_state, low_conf_index, input_images, output_images, predictions_state],
        outputs=[output_images, low_conf_state, predictions_state]
    ).then(
        fn=lambda x, y, z: display_current_annotations(x or [], y, z),  # Chain to update JSON display
        inputs=[low_conf_state, input_images, predictions_state],
        outputs=[live_json]
    ).then(
        fn=create_weld_statistics_chart,  # Update statistics chart after manual edits
        inputs=[predictions_state],
        outputs=[statistics_plot]
    )
    
    # Download button: Export final annotations
    download_button.click(
        fn=export_predictions,
        inputs=[
            project_start_date,
            project_description,
            project_created_by,
            project_status,
            predictions_state  # Add predictions state
        ],
        outputs=[download_file]
    )
    
    # PDF Download button: Generate PDF report
    pdf_download_button.click(
        fn=generate_pdf_report,
        inputs=[
            input_images,
            project_location,
            project_start_date,
            project_description,
            project_created_by,
            project_status,
            project_end_date,
            qa_description,  # Add QA inputs here
            qa_created_by,   # Add QA inputs here
            predictions_state,  # Add predictions state
            project_dropdown   # Add project dropdown value
        ],
        outputs=[pdf_download_file]
    )
    
    project_dropdown.change(
        fn=get_project_details,
        inputs=[project_dropdown],
        outputs=[
            project_location,
            project_start_date,
            project_end_date,
            project_description,
            project_created_by,
            project_status
        ]
    )


if __name__ == "__main__":
    demo.launch(share=True)