# Gamuda Weld Detection Docker Setup

This is a YOLO-based weld detection application with a Gradio web interface that can be easily deployed using Docker.

## Quick Start

### Using Docker Compose (Recommended)

1. Build and run the application:
```bash
docker compose up --build
```

2. Access the application at: http://localhost:7860

### Using Docker directly

1. Build the Docker image:
```bash
docker build -t gamuda-weld-detection .
```

2. Run the container:
```bash
docker run -p 7860:7860 gamuda-weld-detection
```

3. Access the application at: http://localhost:7860

## Features

- Single image weld detection
- Batch processing of multiple images
- Adjustable confidence and IoU thresholds
- Real-time detection results with bounding boxes
- Detailed detection information and statistics

## Requirements

- Docker
- Docker Compose (optional, but recommended)

## Application Structure

- `gamuda_demo.py` - Main Gradio application
- `best.pt` - Trained YOLO model weights
- `requirements.txt` - Python dependencies
- `gamuda_train_config.yaml` - Training configuration
- `gamuda_weld.yaml` - Dataset configuration

## Stopping the Application

### Docker Compose
```bash
docker compose down
```

### Docker
```bash
docker stop <container_id>
```
