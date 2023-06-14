import cv2
import torch
import numpy as np
import pyrealsense2 as rs
import math
from decimal import Decimal, ROUND_HALF_UP

# Load the YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', 'bestpup.pt', force_reload=True)
model = torch.hub.load('yolov5', 'custom', 'bestpup.pt', source='local')
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

def find_plane(points):
    print("1")
    c = np.mean(points, axis=0)
    print("2")
    r0 = points - c
    print("3")
    u, s, v = np.linalg.svd(r0)
    print("u:",u)
    print("before")
    nv = v[-1, :]
    ds = np.dot(points, nv)
    param = np.r_[nv, -np.mean(ds)]
    print("last")
    return param



# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the camera stream
profile = pipeline.start(config)

fx = 615.0  # Focal length in x-direction
fy = 615.0  # Focal length in y-direction
cx = 320.0  # Principal point x-coordinate
cy = 240.0  # Principal point y-coordinate

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


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

            # Only draw bounding box and label if confidence is greater than 0.7
            if conf > 0.5:
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(frame, f'{conf}', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#                 cv2.putText(frame, f'{cls}', (int(xyxy[0])+10, int(xyxy[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                start_x = int(xyxy[0])
                start_y = int(xyxy[1])
                end_x = int(xyxy[2])
                end_y = int(xyxy[3])  
                centroid_x, centroid_y = calculate_image_centroid(start_x, start_y, end_x, end_y)
                #print("Centroid:{},{},{},{},{},{}".format(start_x, start_y, end_x, end_y, centroid_x, centroid_y))
#                 distance = depth_frame.get_distance(int(centroid_x), int(centroid_y))
                cv2.putText(frame, f'{cls}', (int(xyxy[2]+10), int(xyxy[3]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                cv2.circle(frame, (int(centroid_x), int(centroid_y)), 10, (0, 0, 255), 2)
                depth_frame = pipeline.wait_for_frames().get_depth_frame()
                # Assuming you have already performed object detection using YOLO and obtained the bounding box coordinates (xmin, ymin, xmax, ymax)
                theta=0

               
                # Get the depth frame from the RealSense camera
                

                # Map the pixel coordinates to 3D world coordinates
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
                depth = depth_frame.get_distance(int(centroid_x), int(centroid_y))
                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [centroid_x, centroid_y], depth)
                print("DS:",depth_scale)
                # Extract the distance (z-coordinate) from the 3D world coordinates
                distance = point[2] * depth_scale

                # Print the distance of the object
                print("Distance to object: {} meters".format(distance))
                
                center_coordinates_array = []
                center_coordinates_array.append([centroid_x, centroid_y])

                if(len(center_coordinates_array) > 0):
                    for i in range(len(center_coordinates_array)):
                        dist = depth_frame.get_distance(int(center_coordinates_array[i][0]), int(center_coordinates_array[i][1]))*1000 #convert to mm

                        #calculate real world coordinates
                        Xtemp = dist*(center_coordinates_array[i][0] -intr.ppx)/intr.fx
                        Ytemp = dist*(center_coordinates_array[i][1] -intr.ppy)/intr.fy
                        Ztemp = dist

                        Xtarget = Xtemp - 35 #35 is RGB camera module offset from the center of the realsense
                        Ytarget = -(Ztemp*math.sin(theta) + Ytemp*math.cos(theta))
                        Ztarget = Ztemp*math.cos(theta) + Ytemp*math.sin(theta)

                        coordinates_text = "(" + str(Decimal(str(Xtarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                                        ", " + str(Decimal(str(Ytarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                                        ", " + str(Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + ")"

                        print("x, y : " + str(int(center_coordinates_array[i][0])))





                 # Print the 3D world coordinates
                
                cv2.putText(frame, f'{distance}', (int(xyxy[2]), int(xyxy[3]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                print("first")
                offset_x = int((xyxy[2] - xyxy[0])/10)
                offset_y = int((xyxy[3] - xyxy[1])/10)
                interval_x = int((xyxy[2] - xyxy[0] -2 * offset_x)/2)
                interval_y = int((xyxy[3] - xyxy[1] -2 * offset_y)/2)
                points = np.zeros([9,3])
                print("second")
                for i in range(3):
                    for j in range(3):
                        x = int(xyxy[0]) + offset_x + interval_x*i
                        y = int(xyxy[1]) + offset_y + interval_y*j
                        dist = depth_frame.get_distance(x, y)*1000
                        Xtemp = dist*(x - intr.ppx)/intr.fx
                        Ytemp = dist*(y - intr.ppy)/intr.fy
                        Ztemp = dist
                        points[j+i*3][0] = Xtemp
                        points[j+i*3][1] = Ytemp
                        points[j+i*3][2] = Ztemp
                print("third")
                print(points)
                param = find_plane(points)
                print("fourth")
                alpha = math.atan(param[2]/param[0])*180/math.pi
                if(alpha < 0):
                    alpha = alpha + 90
                else:
                    alpha = alpha - 90

                gamma = math.atan(param[2]/param[1])*180/math.pi
                if(gamma < 0):
                    gamma = gamma + 90
                else:
                    gamma = gamma - 90

                text1 = "alpha : " + str(round(alpha))
                text2 = "gamma : " + str(round(gamma))
                cv2.putText(frame, text1, (int((xyxy[0] + xyxy[2])/2) - 40, int((xyxy[1] + xyxy[3])/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(frame, text2, (int((xyxy[0] + xyxy[2])/2) - 40, int((xyxy[1] + xyxy[3])/2) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                cv2.putText(frame, coordinates_text, (int((xyxy[0] + xyxy[2])/2) - 40, int((xyxy[1] + xyxy[3])/2) + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                
                print("alpha : " + str(alpha) + ", gamma : " + str(gamma))


                
    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Stop the camera stream
pipeline.stop()

# Close all windows-
cv2.destroyAllWindows()
