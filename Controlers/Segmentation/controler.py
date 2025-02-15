from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import shutil
import os
import requests
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
from typing import List, Dict
import tempfile
from pydantic import BaseModel
from models.models import ImageURL, AnalysisResponse
import torch
from models.models import  ClassificationCounts
import torchvision.transforms as transforms
import torchvision.models as models
model = None
clafication_model = None

def load_model():
    global model
    global clafication_model
    try:
        current_dir = Path(__file__).parent.resolve()
        
        model_path = current_dir.parent.parent / "models" / "segmentation1.pt"
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model") 
     
     
def download_image(url: str) -> Path:
    """Download image from URL and save to temporary file"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Verify content type is image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")
        
        # Save to temporary file
        suffix = '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
            return Path(tmp.name)
    except requests.RequestException as e:
        raise ValueError(f"Failed to download image: {str(e)}")   


async def analyze_products(image_url: ImageURL):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        temp_image_path = download_image(str(image_url.url))
        results = model.predict(source=str(temp_image_path), save=False, conf=0.25)
        
        # Get shelf analysis
        shelves_data = analyze_shelves(results)
        
        # Calculate totals
        total_ramy = sum(shelf['ramy_count'] for shelf in shelves_data)
        total_products = sum(shelf['total_count'] for shelf in shelves_data)
        
        # Create the response using direct dictionary construction
        response_dict = {
            "total_products": total_products,
            "total_ramy": total_ramy,
            "shelves": [
                {
                    "shelf_number": i + 1,
                    "ramy_count": shelf['ramy_count'],
                    "other_count": shelf['other_count']
                }
                for i, shelf in enumerate(shelves_data)
            ]
        }
        
        # Return the dictionary directly - FastAPI will handle the validation
        return response_dict

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if 'temp_image_path' in locals():
            temp_image_path.unlink(missing_ok=True)



def analyze_shelves(results, num_shelves=3):
    """
    Analyze detection results to identify products in a fixed number of shelves.

    Args:
        results: YOLOv8 detection results
        num_shelves: Number of shelves to divide products into

    Returns:
        list: List of shelf dictionaries containing product counts
    """
    result = results[0] if isinstance(results, list) else results

    if not hasattr(result, 'boxes') or result.boxes is None:
        return []

    # Extract all bounding boxes and their classes
    boxes = []
    min_y = float('inf')
    max_y = float('-inf')
    
    for i in range(len(result.boxes)):
        box = result.boxes[i]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0])
        center_y = (y1 + y2) / 2
        
        # Update min and max y coordinates
        min_y = min(min_y, center_y)
        max_y = max(max_y, center_y)
        
        boxes.append({
            'center_y': center_y,
            'class_id': class_id,
            'bbox': (x1, y1, x2, y2)
        })

    # Calculate shelf boundaries
    shelf_height = (max_y - min_y) / num_shelves
    shelf_boundaries = [
        (min_y + i * shelf_height, min_y + (i + 1) * shelf_height)
        for i in range(num_shelves)
    ]

    # Initialize shelves
    shelves = []
    
    # Assign boxes to shelves
    for shelf_idx, (shelf_min, shelf_max) in enumerate(shelf_boundaries):
        shelf_boxes = [
            box for box in boxes 
            if shelf_min <= box['center_y'] < shelf_max
        ]
        
        # Process shelf
        ramy_count = sum(1 for box in shelf_boxes if box['class_id'] == 0)
        other_count = sum(1 for box in shelf_boxes if box['class_id'] == 1)
        
        shelves.append({
            'ramy_count': ramy_count,
            'other_count': other_count,
            'total_count': len(shelf_boxes)
        })

    return shelves


def process_shelf(shelf_boxes):
    """Process boxes in a shelf to count products by type."""
    ramy_count = sum(1 for box in shelf_boxes if box['class_id'] == 0)
    other_count = sum(1 for box in shelf_boxes if box['class_id'] == 1)

    return {
        'ramy_count': ramy_count,
        'other_count': other_count,
        'total_count': len(shelf_boxes)
    }


def load_classification_model():
    global classification_model
    try:
        current_dir = Path(__file__).parent.resolve()
        model_path = current_dir.parent.parent / "models" / "best_model.pth"

        # Define the model architecture
        classification_model = models.resnet18(pretrained=False)  # Use the correct model architecture
        # Load the state_dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        classification_model.load_state_dict(state_dict)

        classification_model.eval()
        print("Classification model loaded successfully!")
        return classification_model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        raise RuntimeError("Failed to load classification model")

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess image for classification model
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension

async def classify_products(image_url: ImageURL):
    """
    Process an image through segmentation and classification
    """
    if not model or not classification_model:
        raise HTTPException(status_code=500, detail="Models not loaded")
    try:
        # Download and process image
        temp_image_path = download_image(str(image_url.url))
        original_image = cv2.imread(str(temp_image_path))
        
        # Run segmentation
        seg_results = model.predict(source=str(temp_image_path), save=False, conf=0.25)
        
        # Extract Ramy product boxes
        ramy_boxes = []
        result = seg_results[0]
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Ramy product
                ramy_boxes.append(box.xyxy[0].cpu().numpy())
        
        # Process each product
        class_counts = {1: 0, 2: 0, 3: 0}
        
        for box in ramy_boxes:
            # Extract and preprocess product image
            x1, y1, x2, y2 = map(int, box)
            product_img = original_image[y1:y2, x1:x2]
            
            # Preprocess for classification
            input_tensor = preprocess_image(product_img)
            
            # Run inference
            with torch.no_grad():
                output = classification_model(input_tensor)
                # Assuming output is logits, convert to probabilities
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                class_counts[predicted_class + 1] += 1
        
        # Prepare response
        response = ClassificationCounts(
            class_1_count=class_counts[1],
            class_2_count=class_counts[2],
            class_3_count=class_counts[3],
            total_ramy_products=len(ramy_boxes)
        )
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    finally:
        if 'temp_image_path' in locals():
            temp_image_path.unlink(missing_ok=True)

