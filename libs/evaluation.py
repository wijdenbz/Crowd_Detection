import os
import time
import numpy as np
from ultralytics import YOLO
import cv2

def get_model_info(model_path):
    """Get model size in MB and number of parameters"""
    try:
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)  # Convert to MB
        model = YOLO(model_path)
        num_params = sum(p.numel() for p in model.parameters())
        return size_mb, num_params
    except:
        return 0, 0

def get_video_orientation(frame):
    """Detect video orientation based on frame dimensions"""
    if frame is None:
        return 'landscape'
    height, width = frame.shape[:2]
    return 'portrait' if height > width else 'landscape'

def ensure_frame_dims(frame):
    """Ensure frame dimensions are valid"""
    if frame is None:
        return None
    try:
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None
        return frame
    except:
        return None

def enhance_frame(frame):
    """Apply image enhancement techniques"""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        return enhanced
    except Exception:
        return frame

def handle_portrait_video(frame, force_rotate=False):
    """
    Handle portrait video orientation
    If force_rotate is True, always rotate to landscape orientation
    """
    if frame is None:
        return None
    
    height, width = frame.shape[:2]
    orientation = get_video_orientation(frame)
    
    # Only rotate if portrait and force_rotate is True
    if orientation == 'portrait' and force_rotate:
        # Rotate 90 degrees clockwise to make it landscape
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return rotated
    
    return frame

def resize_frame(frame, target_size=(640, 640), is_portrait=False, force_rotate=False):
    """Resize frame while maintaining aspect ratio with portrait handling"""
    frame = ensure_frame_dims(frame)
    if frame is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    # Apply portrait handling if needed
    if is_portrait:
        frame = handle_portrait_video(frame, force_rotate)
    
    height, width = frame.shape[:2]
    orientation = 'portrait' if height > width else 'landscape'
    
    # Store original dimensions for later use
    orig_dims = (width, height)
    
    # Calculate target dimensions while maintaining aspect ratio
    if orientation == 'portrait':
        # For portrait videos, use height as the primary dimension
        scale = min(target_size[1] / height, target_size[0] / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
    else:
        # For landscape videos, use width as the primary dimension
        scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    
    # Ensure minimum dimensions
    new_width = max(new_width, 32)
    new_height = max(new_height, 32)
    
    # Resize frame using INTER_LINEAR for better quality
    try:
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    # Create background
    background = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Calculate position to center the frame
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    # Ensure valid dimensions for placement
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)
    
    try:
        # Place resized frame on background
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    return background

def process_frame(frame, model, conf_threshold, classes_to_count, is_portrait=False, force_rotate=False):
    """Process a single frame and return metrics with portrait handling"""
    start_time = time.time()
    
    # Ensure valid frame
    frame = ensure_frame_dims(frame)
    if frame is None:
        return np.zeros((640, 640, 3), dtype=np.uint8), 0, 0
    
    # Store original frame and dimensions
    orig_frame = frame.copy()
    orig_height, orig_width = frame.shape[:2]
    
    # Get video orientation
    orientation = get_video_orientation(frame)
    
    # Handle portrait video if needed
    if is_portrait or orientation == 'portrait':
        # If force_rotate, rotate the original frame
        if force_rotate:
            orig_frame = handle_portrait_video(orig_frame, force_rotate=True)
            # Update dimensions after rotation
            orig_height, orig_width = orig_frame.shape[:2]
    
    # Enhance frame
    enhanced_frame = enhance_frame(orig_frame.copy())
    
    # Resize frame while maintaining aspect ratio (with portrait handling)
    process_frame = resize_frame(enhanced_frame, is_portrait=is_portrait, force_rotate=force_rotate)
    
    try:
        # Run YOLO prediction with improved confidence
        results = model.predict(
            process_frame,
            conf=conf_threshold,
            iou=0.45,  # Improved IOU threshold
            classes=classes_to_count  # Only detect specified classes
        )[0]
        
        # Get detection results
        boxes = results.boxes.xyxy.cpu().numpy()
        classes_detected = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
    except Exception:
        return orig_frame, 0, time.time() - start_time
    
    # Count objects
    object_count = len(classes_detected)
    
    # Draw bounding boxes on the original frame
    display_frame = orig_frame.copy()
    
    # Calculate scale factors for bounding boxes
    scale_x = orig_width / 640
    scale_y = orig_height / 640
    
    # Define green color for all visualizations
    box_color = (0, 255, 0)  # BGR Green
    
    # Draw detection boxes
    for idx, cls in enumerate(classes_detected):
        if int(cls) in classes_to_count:
            try:
                x1, y1, x2, y2 = boxes[idx]
                conf = confidences[idx]
                
                # Scale coordinates to original frame size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, orig_width - 1))
                y1 = max(0, min(y1, orig_height - 1))
                x2 = max(0, min(x2, orig_width - 1))
                y2 = max(0, min(y2, orig_height - 1))
                
                # Draw thinner box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 1)
                
                # Add confidence score with green background
                label = f"{conf:.2f}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                y1 = max(y1, label_size[1])
                
                # Draw green background for text
                cv2.rectangle(display_frame, 
                            (x1, y1 - label_size[1] - baseline),
                            (x1 + label_size[0], y1), 
                            box_color, 
                            cv2.FILLED)
                
                # Draw text in black
                cv2.putText(display_frame, 
                           label, 
                           (x1, y1 - baseline),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4,  # Smaller font size
                           (0, 0, 0), 
                           1)  # Thinner text
            except Exception:
                continue
    
    # Add count to frame with better visibility
    try:
        count_text = f"Count: {object_count}"
        cv2.rectangle(display_frame, (5, 5), (120, 35), box_color, cv2.FILLED)
        cv2.putText(display_frame, 
                    count_text, 
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,  # Smaller font size
                    (0, 0, 0), 
                    1)  # Thinner text
    except Exception:
        pass
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    return display_frame, object_count, inference_time

def calculate_performance_metrics(video_length, frame_size, model_name, num_params, 
                                avg_fps, avg_inference_time, total_inference_time, 
                                total_objects_detected, video_name, total_frames,
                                avg_objects_per_frame, max_objects_detected,
                                is_portrait=False):
    """Calculate and return performance metrics"""
    
    orientation = "Portrait" if is_portrait else "Landscape"
    
    return {
        "Metric": [
            "Video Name",
            "Video Length (frames)",
            "Frame Size", 
            "Orientation",
            "Model", 
            "Parameters",
            "Average FPS",
            "Inference Time per Frame (ms)", 
            "Average Objects per Frame",
            "Max Objects Detected"
        ],
        "Value": [
            video_name,
            total_frames,
            f"{frame_size[0]}x{frame_size[1]}",
            orientation,
            model_name,
            f"{num_params:,}",
            f"{avg_fps:.2f}",
            f"{avg_inference_time * 1000:.2f}",
            f"{avg_objects_per_frame:.2f}",
            max_objects_detected
        ]
    }