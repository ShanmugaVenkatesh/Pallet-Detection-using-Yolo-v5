import cv2
import torch
import numpy as np
import pyrealsense2 as rs

# Load the YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best_upp(11).pt', force_reload=True)
model = torch.hub.load('yolov5', 'custom', 'best_upp(11).pt', source='local')
model.to(device).eval()

# Set camera parameters
width, height = 640, 480  # Set the desired frame size
fps = 30  # Set the desired frame rate

def calculate_image_centroid(start_x, start_y, end_x, end_y):
    width = end_x - start_x
    height = end_y - start_y
    centroid_x = start_x + (width / 2)
    centroid_y = start_y + (height / 2)
    return centroid_x, centroid_y
def calculate_box_dimensions(xyxy):
    x1, y1, x2, y2 = xyxy
    width = x2 - x1
    height = y2 - y1
    return width, height
# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the camera stream
pipeline.start(config)

# Object detection loop
while True:
    # Wait for a new frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Convert the frame to a numpy array
    frame = np.asanyarray(color_frame.get_data())
    d_frame = np.asanyarray(depth_frame.get_data())

    # Perform object detection
    results = model(frame)
    print(results)
    # Display the results
    for result in results.xyxy[0]:
        if result is not None:
            xyxy = result[:4].tolist()
            conf = result[4].item()
            cls = int(result[5].item())
            box_width, box_height = calculate_box_dimensions(xyxy)
            #print(f"Width: {box_width}, Height: {box_height}")
                    

            # Only draw bounding box and label if confidence is greater than 0.7
            if conf > 0.5:
                # Draw bounding box and label on the frame
                if cls == 0 and box_width >=230 and box_height >=120 :
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'W: {box_width}', (int(xyxy[2]), int(xyxy[3]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f'H: {box_height}', (int(xyxy[2]), int(xyxy[3]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    
                else:
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                    
                #cv2.putText(frame, f'{conf}', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#                 cv2.putText(frame, f'{cls}', (int(xyxy[0])+10, int(xyxy[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                start_x = int(xyxy[0])
                start_y = int(xyxy[1])
                end_x = int(xyxy[2])
                end_y = int(xyxy[3])
                centroid_x, centroid_y = calculate_image_centroid(start_x, start_y, end_x, end_y)
                #print("Centroid:{},{},{},{},{},{}".format(start_x, start_y, end_x, end_y, centroid_x, centroid_y))
                distance = depth_frame.get_distance(int(centroid_x), int(centroid_y))
               
                cv2.putText(frame, f'{cls}', (int(xyxy[2]+10), int(xyxy[3]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                if cls == 0:
                    cv2.circle(frame,(int(centroid_x),int(centroid_y)),10,(0,0,255),2)
#                 cv2.circle(frame, (int(centroid_x), int(centroid_y)), 10, (0, 0, 255), 2)
    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Stop the camera stream
pipeline.stop()

# Close all windows
cv2.destroyAllWindows()