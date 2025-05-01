import cv2
import numpy as np

def detect_orientation(frame):
    """
    Detect if a frame is in portrait or landscape orientation.
    
    Args:
        frame: OpenCV image frame
        
    Returns:
        str: 'portrait' or 'landscape'
    """
    if frame is None:
        return 'landscape'
        
    height, width = frame.shape[:2]
    return 'portrait' if height > width else 'landscape'

def check_video_orientation(video_path):
    """
    Check the orientation of a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        str: 'portrait' or 'landscape'
        tuple: (width, height) of the video
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        return 'landscape', (0, 0)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    orientation = detect_orientation(frame)
    cap.release()
    
    return orientation, (width, height)

def rotate_frame_to_landscape(frame):
    """
    Rotate a portrait frame to landscape orientation.
    
    Args:
        frame: OpenCV image frame
        
    Returns:
        ndarray: Rotated frame (90 degrees clockwise)
    """
    if frame is None:
        return None
        
    orientation = detect_orientation(frame)
    
    if orientation == 'portrait':
        # Rotate 90 degrees clockwise
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # Already landscape, no rotation needed
    return frame

def rotate_frame_to_portrait(frame):
    """
    Rotate a landscape frame to portrait orientation.
    
    Args:
        frame: OpenCV image frame
        
    Returns:
        ndarray: Rotated frame (90 degrees counter-clockwise)
    """
    if frame is None:
        return None
        
    orientation = detect_orientation(frame)
    
    if orientation == 'landscape':
        # Rotate 90 degrees counter-clockwise
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Already portrait, no rotation needed
    return frame

def resize_keeping_aspect_ratio(frame, target_size=(640, 640)):
    """
    Resize a frame to target size while preserving aspect ratio.
    Places the resized image on a black background of target_size.
    
    Args:
        frame: OpenCV image frame
        target_size: Tuple of (width, height) for target size
        
    Returns:
        ndarray: Resized frame on black background
    """
    if frame is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    # Get dimensions
    height, width = frame.shape[:2]
    
    # Calculate scale to maintain aspect ratio
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Ensure minimum dimensions
    new_width = max(new_width, 32)
    new_height = max(new_height, 32)
    
    # Resize frame
    try:
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    # Create black background
    background = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Calculate position to center the frame
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    # Place resized frame on background
    try:
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    return background

def process_portrait_video(frame, force_rotate=False, target_size=(640, 640)):
    """
    Process a frame with portrait video handling.
    
    Args:
        frame: OpenCV image frame
        force_rotate: Whether to force rotation to landscape
        target_size: Target size for processing
        
    Returns:
        ndarray: Processed frame
        tuple: Original dimensions (height, width)
    """
    if frame is None:
        return np.zeros((*target_size, 3), dtype=np.uint8), (0, 0)
    
    # Store original dimensions
    orig_dims = frame.shape[:2]  # (height, width)
    
    # Get orientation
    orientation = detect_orientation(frame)
    
    # Apply rotation if needed
    if orientation == 'portrait' and force_rotate:
        frame = rotate_frame_to_landscape(frame)
    
    # Resize while maintaining aspect ratio
    processed = resize_keeping_aspect_ratio(frame, target_size)
    
    return processed, orig_dims