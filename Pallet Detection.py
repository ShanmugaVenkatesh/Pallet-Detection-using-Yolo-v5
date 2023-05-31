import cv2
import torch
import numpy as np
import pyrealsense2 as rs

# Load the YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model = torch.hub.load('yolov5', 'custom', path='best.pt',source='local')
model.to(device).eval()

# Set camera parameters
width, height = 640, 480  # Set the desired frame size
fps = 30  # Set the desired frame rate

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

# Start the camera stream
pipeline.start(config)

try:
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert the frame to a numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Perform object detection
        results = model(frame)

        # Display the results
        for result in results.xyxy[0]:
            if result is not None:
                xyxy = result[:4].tolist()
                conf = result[4].item()
                cls = int(result[5].item())

                # Only draw bounding box and label if confidence is greater than 0.3
                if conf > 0.6:
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, f'{conf}', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 0, 0), 2)

                    # Extract the region within the bounding box
                    object_region = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

                    # Convert the region to grayscale
                    gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)

                    # Apply Gaussian blur to reduce noise
                    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

                    # Perform Canny edge detection
                    edges = cv2.Canny(blurred, 30, 100)  # Adjust the threshold values as needed

                    # Display the edges within the bounding box
                    frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Stop the camera stream
    pipeline.stop()

    # Close all windows
    cv2.destroyAllWindows()
