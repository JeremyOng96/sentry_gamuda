import json
import os
from datetime import datetime
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import tempfile
import glob
import logging
from utils.image_utils import draw_boxes, hex_to_rgb
from database.db_utils import get_specific_results, get_reports_from_database

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class WeldReportGenerator:
    logger = logging.getLogger(__name__)
    
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

        # PDF related colors
        self.text_colors = {
            'good_weld': (0, 255, 0),  # Green
            'bad_weld': (255, 0, 0),   # Red
            'header': (41, 128, 185),  # Blue
            'text': (52, 73, 94),      # Dark gray
            'background': (236, 240, 241)  # Light gray
        }
        
        # Grid layout settings
        self.images_per_row = 4
        self.images_per_page = 20  # 4x3 grid (3 rows of 4 images)
        self.max_images = 30  # Maximum images to show in report
        self.image_width = 45  # mm (larger to fit 4 per row)
        self.image_height = 34  # mm (proportionally larger)
        self.margin = 4  # mm
        
    def add_header_page(self, metadata: Dict[str, Any], annotations: List[Dict]):
        """Add a header page with project information and summary statistics"""
        self.pdf.add_page()
        
        # Header text without gray background
        # Left header text
        self.pdf.set_font('Arial', '', 10)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.set_xy(10, 10)
        self.pdf.cell(0, 5, 'Gamuda IBS', 0, 0, 'L')
        
        # Right header text
        self.pdf.set_xy(150, 10)
        self.pdf.cell(50, 5, 'Form detail report', 0, 0, 'R')
        
        # Add line below header text
        self.pdf.set_draw_color(200, 200, 200)  # Light gray line
        self.pdf.line(10, 18, 200, 18)  # Horizontal line from left to right margin
        
        self.colors = metadata.get('colors', [])
        self.labels = metadata.get('labels', [])
        # Add logo if exists (top right) - scale according to aspect ratio
        logo_path = "/Users/jeremyong/Desktop/Parexus/Gamuda/gamuda-logo-header-red.png"
        if os.path.exists(logo_path):
            self.pdf.image(logo_path, x=160, y=20, w=40)
        
        # Form detail section (minimal spacing after header line)
        self.pdf.ln(8)  # Much closer to header line
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.set_text_color(128, 128, 128)
        self.pdf.cell(0, 8, 'Form detail', ln=True)
        
        # Main title with form number
        self.pdf.set_font('Arial', 'B', 20)
        self.pdf.set_text_color(41, 128, 185)
        reports = get_reports_from_database()
        self.num_reports = len(reports) if reports else 0

        form_id = self.num_reports + 1
        report_title = metadata.get('report_title', 'Defect Management Form')
        self.pdf.cell(0, 12, f'#{form_id}: {report_title}', ln=True, align='L')
        
        # Forms section header
        self.pdf.ln(20)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.set_text_color(80, 80, 80)
        self.pdf.cell(0, 10, 'Forms', ln=True)
        
        # Add horizontal line under "Forms"
        self.pdf.set_draw_color(200, 200, 200)
        current_y = self.pdf.get_y()
        self.pdf.line(10, current_y, 200, current_y)
        
        # Project details with better spacing
        self.pdf.set_font('Arial', '', 11)
        self.pdf.set_text_color(80, 80, 80)
        
        project_info = [
            ('Location', metadata.get('location', 'CO-11')),
            ('Form date', metadata.get('date', 'Aug 14, 2025')),
            ('Template', metadata.get('template', 'Defect Management Form')),
            ('Description', metadata.get('description', 'Welding Anchor Plate Not Align')),
            ('Due date', metadata.get('due_date', 'Not specified')),
            ('Created by', metadata.get('created_by')),
            ('Status', metadata.get('status', 'In progress')),
            ('QA Description', metadata.get('qa_description', 'Not specified')),
            ('QA Created by', metadata.get('qa_created_by', 'Not specified')),
            ('Report generated at', metadata.get('export_time')),
        ]
        
        # Set border color for project info table
        self.pdf.set_draw_color(220, 220, 220)  # Light gray border color
        
        for label, value in project_info:
            self.pdf.set_font('Arial', '', 11)
            self.pdf.set_text_color(120, 120, 120)
            self.pdf.cell(50, 12, label, 'B', 0, 'L')  # Label column
            self.pdf.set_text_color(80, 80, 80)
            self.pdf.cell(0, 12, str(value), 'B', 1, 'L')  # Value column

        # Add Detection Summary Table on page 1
        self.pdf.ln(15)
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.set_text_color(41, 128, 185)
        self.pdf.cell(0, 10, 'Detection Summary', ln=True, align='L')
        self.pdf.ln(2)
        
        # Get statistics from metadata
        try:
            stats = metadata.get('statistics', {})
            good_welds = stats.get('good_welds', 0)
            bad_welds = stats.get('bad_welds', 0)
            edited_welds = stats.get('edited_welds', 0)
            total_detections = stats.get('total_welds', 0)
            
            self.logger.info(f"Report statistics - Good: {good_welds}, Bad: {bad_welds}, Edited: {edited_welds}, Total: {total_detections}")
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            total_images = total_detections = good_welds = bad_welds = edited_welds = 0
        
        report_date = metadata.get('export_time', 'Not specified')
        # Summary table with new column structure
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_text_color(52, 73, 94)
        
        # Table headers - 5 columns as requested
        col_width = 36  # Smaller columns to fit 5 columns
        self.pdf.cell(col_width, 10, 'Date', 1, 0, 'C')
        self.pdf.cell(col_width, 10, 'Good Weld', 1, 0, 'C')
        self.pdf.cell(col_width, 10, 'Bad Weld', 1, 0, 'C')
        self.pdf.cell(col_width, 10, 'Human Edited', 1, 0, 'C')
        self.pdf.cell(col_width, 10, 'Total Welds', 1, 1, 'C')

        # Get historical results
        project_id = metadata.get('project_id')
        created_by = metadata.get('qa_created_by') or metadata.get('created_by')
        historical_results = get_specific_results(project_id, created_by)
        logger.info(f"Historical results: {historical_results}")
        # Combine current and historical results
        all_results = []

        # Add historical results, only retrieve the 4 latest results        
        for result in historical_results[:4]:
            total = result['total_welds']
            if total > 0:
                all_results.append({
                    'date': result['created_at'],
                    'good_weld': result['good_weld'],
                    'bad_weld': result['bad_weld'],
                    'uncertain': result['uncertain'],
                    'edited': 0,  # Historical records don't track edits
                    'total': total
                })
        
        # Add current result
        if total_detections > 0:
            all_results.append({
                'date': report_date,
                'good_weld': good_welds,
                'bad_weld': bad_welds,
                'uncertain': stats.get('low_confidence', 0),
                'edited': edited_welds,
                'total': total_detections
            })
        
        # Sort results by date (newest first)
        all_results.sort(key=lambda x: x['date'], reverse=True)  # reverse=True for descending order
        
        # Table data r#ows
        self.pdf.set_font('Arial', '', 9)
        
        for result in all_results:
            total = result['total']
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.cell(col_width, 10, result['date'], 1, 0, 'C')
            self.pdf.cell(col_width, 10, f"{(result['good_weld']/total*100):.1f}%", 1, 0, 'C')
            self.pdf.cell(col_width, 10, f"{(result['bad_weld']/total*100):.1f}%", 1, 0, 'C')
            self.pdf.cell(col_width, 10, f"{(result['uncertain']/total*100):.1f}%", 1, 0, 'C')
            self.pdf.cell(col_width, 10, str(total), 1, 1, 'C')
        
        self.pdf.ln(10)
        

    def draw_bounding_boxes_on_image(self, image_path: str, image_id: int, annotations: List[Dict], categories: List[Dict], output_path: str) -> str:
        """Draw bounding boxes on image and save to temporary file"""
        image = cv2.imread(image_path)

        image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
        for annotation in image_annotations:
            color = self.colors[annotation["category_id"]]
            bbox = annotation["bbox"]
            category_name = self.labels[annotation["category_id"]]
            confidence = annotation["confidence"]
            image = draw_boxes(image, [bbox], [confidence], [category_name], [color])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, image)
        return output_path
    
    def add_image_grid_page(self, images_data: List[Dict], image_directory: str, page_num: int, annotations: List[Dict], categories: List[Dict]):
        """Add a page with a 4x5 grid of images"""
        self.pdf.add_page()
        
        # Add logo if exists (right aligned) - scale according to aspect ratio
        logo_path = "/Users/jeremyong/Desktop/Parexus/Gamuda/gamuda-logo-header-red.png"
        if os.path.exists(logo_path):
            self.pdf.image(logo_path, x=160, y=5, w=40)
        
        # Page header
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.set_text_color(41, 128, 185)
        self.pdf.ln(25)  # Move down from header
        self.pdf.cell(0, 10, f'Weld Detection Results Sample - Page {page_num}', ln=True, align='C')
        self.pdf.ln(5)
        
        # Calculate positions for 4x3 grid
        start_x = 12
        start_y = 45  # Adjusted for header (25 + 20)
        spacing_x = 48  # Spacing for 4 images per row
        spacing_y = 45  # Vertical spacing
        
        temp_files = []  # Keep track of temporary files for cleanup
        
        for i, image_data in enumerate(images_data):
            if i >= self.images_per_page:
                break
                
            row = i // self.images_per_row
            col = i % self.images_per_row
            
            x = start_x + col * spacing_x
            y = start_y + row * spacing_y
            
            # Find the image file
            image_name = image_data['file_name']  # COCO format uses file_name
            image_path = None
            
            # Look for the image in the directory
            possible_paths = [
                os.path.join(image_directory, image_name),
                os.path.join(image_directory, image_name.replace('.jpg', '.png')),
                os.path.join(image_directory, image_name.replace('.png', '.jpg')),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if image_path and os.path.exists(image_path):
                # Create temporary file for annotated image
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                temp_file.close()
                temp_files.append(temp_file.name)
                
                # Draw bounding boxes on image
                annotated_path = self.draw_bounding_boxes_on_image(
                    image_path, image_data['id'], annotations, categories, temp_file.name
                )
                
                # Add image to PDF
                try:
                    self.pdf.image(annotated_path, x=x, y=y, w=self.image_width, h=self.image_height)
                except Exception as e:
                    print(f"Error adding image to PDF: {e}")
                    # Add placeholder rectangle
                    self.pdf.rect(x, y, self.image_width, self.image_height)
                
                # Add image name below image
                self.pdf.set_font('Arial', '', 7)  # Slightly larger font for 4 per row
                self.pdf.set_text_color(52, 73, 94)
                text_y = y + self.image_height + 2
                
                # Truncate long filenames
                display_name = image_name
                if len(display_name) > 20:
                    display_name = display_name[:17] + "..."
                
                # Center text under image
                text_width = self.pdf.get_string_width(display_name)
                text_x = x + (self.image_width - text_width) / 2
                self.pdf.ln(5)
                self.pdf.text(text_x, text_y, display_name)
                
            else:
                # Add placeholder for missing image
                self.pdf.set_fill_color(236, 240, 241)
                self.pdf.rect(x, y, self.image_width, self.image_height, 'F')
                
                # Add "Image not found" text
                self.pdf.set_font('Arial', '', 10)
                self.pdf.set_text_color(128, 128, 128)
                text = "Image not found"
                text_width = self.pdf.get_string_width(text)
                text_x = x + (self.image_width - text_width) / 2
                text_y = y + self.image_height / 2
                self.pdf.text(text_x, text_y, text)
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    
    def generate_report(self, json_file_path: str, image_directory: str, output_path: str = None) -> str:
        """Generate the complete PDF report"""
        # Load JSON data
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])
        
        # Limit to first 30 images
        limited_images = images[:self.max_images]
        image_ids = [img["id"] for img in limited_images]
        
        # Get annotations for these images
        limited_annotations = [
            ann for ann in annotations
            if ann["image_id"] in image_ids
        ]
        
        # Add header page with detection summary
        self.add_header_page(metadata, annotations)
        
        # Add image grid pages (limited to 30 images)
        page_num = 1
        for i in range(0, len(limited_images), self.images_per_page):
            page_images = limited_images[i:i + self.images_per_page]
            self.add_image_grid_page(page_images, image_directory, page_num, limited_annotations, categories)
            page_num += 1
        
        # Generate output filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"weld_detection_report_{timestamp}.pdf"
        
        # Save PDF
        self.pdf.output(output_path)
        return output_path

def generate_report_from_predictions(all_predictions: List[Dict], files: List, metadata: Dict = None, output_path: str = None) -> str:
    """
    Generate a PDF report directly from prediction data and uploaded files
    
    Args:
        all_predictions: List of prediction dictionaries from the model
        files: List of uploaded file objects from Gradio
        metadata: Optional metadata dictionary
        output_path: Optional output path for the PDF report
    
    Returns:
        Path to the generated PDF report
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting report generation from predictions...")
    
    try:
        # Create temporary JSON file from predictions
        import tempfile
        
        # Limit to first 30 images and their annotations
        max_images = 30
        image_ids = [img["id"] for img in all_predictions["images"][:max_images]]
        
        # Get annotations for these images
        annotations = [
            ann for ann in all_predictions["annotations"]
            if ann["image_id"] in image_ids
        ]
        
        # Get limited images
        limited_files = files[:max_images]
        limited_images = all_predictions["images"][:max_images]
        
        logger.info(f"Processing {len(limited_images)} images and {len(annotations)} annotations")
        
        # Create export data structure
        export_data = {
            "metadata": metadata or {
                "location": "Not specified",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "description": "Weld detection report",
                "created_by": "Gamuda Weld Detection System",
                "status": "Completed",
                "export_time": datetime.now().strftime("%Y-%m-%d"),
                "total_images": len(limited_images),
                "total_predictions": len(annotations)
            },
            "images": limited_images,
            "annotations": annotations,
            "categories": all_predictions["categories"]
        }
        
        logger.info("Creating temporary files...")
        
        # Create temporary JSON file
        temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(export_data, temp_json, indent=2)
        temp_json.close()
        logger.info(f"Temporary JSON file created: {temp_json.name}")
        
        # Create temporary directory for images
        temp_image_dir = tempfile.mkdtemp()
        logger.info(f"Temporary image directory created: {temp_image_dir}")
        
        # Copy uploaded files to temporary directory
        for i, file in enumerate(files):
            try:
                if hasattr(file, 'name'):
                    # Read the file content
                    with open(file.name, 'rb') as f:
                        content = f.read()
                    
                    # Write to temp directory
                    filename = os.path.basename(file.name)
                    temp_image_path = os.path.join(temp_image_dir, filename)
                    with open(temp_image_path, 'wb') as f:
                        f.write(content)
                    logger.info(f"Successfully copied file {filename} to temp directory")
                else:
                    logger.warning(f"File at index {i} has no name attribute")
            except Exception as e:
                logger.error(f"Error copying file {i}: {e}")
                continue
        
        try:
            # Generate report
            logger.info("Initializing WeldReportGenerator...")
            generator = WeldReportGenerator()
            logger.info("Generating report...")
            report_path = generator.generate_report(temp_json.name, temp_image_dir, output_path)
            logger.info(f"Report generated successfully: {report_path}")
            
            return report_path
        finally:
            # Clean up temporary files
            try:
                logger.info("Cleaning up temporary files...")
                os.unlink(temp_json.name)
                import shutil
                shutil.rmtree(temp_image_dir)
                logger.info("Temporary files cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
    
    except Exception as e:
        logger.error(f"Error in generate_report_from_predictions: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


# Example usage
if __name__ == "__main__":
    # Example usage
    json_file = "/Users/jeremyong/Desktop/Parexus/Gamuda/all_weld_predictions_20250930_010215.json"
    image_dir = "/Users/jeremyong/Downloads/Weld Detection.v5i.yolov11/test/images"
    
    if os.path.exists(json_file):
        report_path = generate_weld_report(json_file, image_dir)
        print(f"Report generated: {report_path}")
    else:
        print("JSON file not found. Please provide the correct path.")
