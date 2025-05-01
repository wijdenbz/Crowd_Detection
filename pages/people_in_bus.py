import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
import pandas as pd
from libs.evaluation import get_model_info, process_frame, calculate_performance_metrics

# Hide sidebar and other default UI elements
st.markdown("""
    <style>
    /* Hide the page navigation links in the sidebar */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Hide the sidebar toggle button (hamburger icon) */
    [data-testid="collapsedControl"] {
        display: none;
    }

    /* Optional: hide header/footer if needed */
    header, footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👥 People in Bus Detection")

# Sidebar: Upload video
uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="sidebar")

# Sidebar: Model selection
st.sidebar.title("🔧 Settings")

# Only YOLOv11 models as specified
available_models = [
    "yolo11n.pt", 
    "yolo11s.pt", 
    "yolo11m.pt", 
    "yolo11l.pt", 
    "yolo11x.pt"
]

selected_model = st.sidebar.selectbox("Select Model", available_models)

# Classes to count (0=person by default)
classes_to_count = [0]  # Person class in COCO dataset

# Confidence threshold slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

# Additional settings for improved detection
st.sidebar.title("🎯 Detection Settings")
iou_threshold = st.sidebar.slider("IOU Threshold", 0.1, 0.9, 0.45, 0.05)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

# Main app behavior
if uploaded_file is not None:
    input_video_path = "temp_video.mp4"
    with open(input_video_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Create two columns with equal width for videos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Original Video")
        # Get video dimensions for consistent display
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Display original video
        st.video(input_video_path)
    
    if st.button("🚀 Start Processing"):
        # Load model
        with st.spinner(f"Loading model {selected_model}..."):
            try:
                model = YOLO(selected_model)
                model_size, num_params = get_model_info(selected_model)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()
        
        # Initialize video capture
        cap = cv2.VideoCapture(input_video_path)
        assert cap.isOpened(), "Error reading video file"
        
        # Get video properties
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a text placeholder for progress updates
        progress_text = st.empty()
        
        # Create a frame placeholder for live preview (same size as original)
        with col2:
            st.subheader("🎬 Live Processing")
            frame_placeholder = st.empty()
        
        # Create metrics tracking
        frame_count = 0
        total_inference_time = 0
        total_objects_detected = 0
        fps_list = []
        object_counts = []
        
        # Set up metrics display
        metrics_placeholder = st.empty()
        
        # Process video
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame and get metrics
            processed_frame, count, inference_time = process_frame(frame, model, conf_threshold, classes_to_count)
            
            # Update metrics
            frame_count += 1
            total_inference_time += inference_time
            total_objects_detected += count
            object_counts.append(count)
            
            # Calculate FPS
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_list.append(current_fps)
            
            # Update display every 5 frames to avoid slowdown
            if frame_count % 5 == 0 or frame_count == total_frames:
                # Display processed frame
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                
                # Update progress text
                progress_percentage = (frame_count / total_frames) * 100
                progress_text.markdown(f"### Processing: {frame_count}/{total_frames} frames ({progress_percentage:.2f}%)")
                
                # Display real-time metrics
                avg_fps = sum(fps_list[-20:]) / min(len(fps_list), 20)  # Rolling average
                avg_inference_time = total_inference_time / frame_count
                avg_objects_per_frame = total_objects_detected / frame_count if frame_count > 0 else 0
                max_objects_detected = max(object_counts) if object_counts else 0
                
                metrics_data = calculate_performance_metrics(
                    video_length=total_frames / fps,
                    frame_size=(w, h),
                    model_name=selected_model,
                    num_params=num_params,
                    avg_fps=avg_fps,
                    avg_inference_time=avg_inference_time,
                    total_inference_time=total_inference_time,
                    total_objects_detected=total_objects_detected,
                    video_name=uploaded_file.name,
                    total_frames=total_frames,
                    avg_objects_per_frame=avg_objects_per_frame,
                    max_objects_detected=max_objects_detected
                )
                metrics_placeholder.dataframe(pd.DataFrame(metrics_data), hide_index=True)
        
        # Release resources
        cap.release()
        
        # Final success message
        st.success(f"✅ Processing completed! Processed {frame_count} frames.")
        
        # Display performance data in two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display summary metrics
            avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
            
            # Create performance table
            st.subheader("📊 Performance Metrics")
            
            avg_objects_per_frame = total_objects_detected / frame_count if frame_count > 0 else 0
            max_objects_detected = max(object_counts) if object_counts else 0
            
            results_table = pd.DataFrame(calculate_performance_metrics(
                video_length=total_frames / fps,
                frame_size=(w, h),
                model_name=selected_model,
                num_params=num_params,
                avg_fps=avg_fps,
                avg_inference_time=avg_inference_time,
                total_inference_time=total_inference_time,
                total_objects_detected=total_objects_detected,
                video_name=uploaded_file.name,
                total_frames=total_frames,
                avg_objects_per_frame=avg_objects_per_frame,
                max_objects_detected=max_objects_detected
            ))
            
            st.dataframe(results_table)
        
        with col2:
            # Display chart of object counts over time
            if frame_count > 0:
                st.subheader("📈 Detection Metrics")
                
                # Create tabs for different charts
                tab1, tab2 = st.tabs(["FPS", "Object Count"])
                
                with tab1:
                    fps_chart_data = pd.DataFrame({
                        'frame': list(range(1, frame_count + 1)),
                        'FPS': fps_list
                    })
                    st.line_chart(fps_chart_data.set_index('frame'))
                
                with tab2:
                    count_chart_data = pd.DataFrame({
                        'frame': list(range(1, frame_count + 1)),
                        'Count': object_counts
                    })
                    st.line_chart(count_chart_data.set_index('frame'))

else:
    st.info("👈 Please upload a video file to start.")
