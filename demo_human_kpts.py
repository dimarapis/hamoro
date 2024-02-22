import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt


model = YOLO("yolov8l-pose.pt")  # load a pretrained model (recommended for training)

def pose(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    results = model(image, verbose=True)
    annotated_frame = results[0].plot()
    return annotated_frame

def l515():
    # Initialize the RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    try:
        while True:
            # Wait for a new frame from the camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert RealSense frame to OpenCV format
            image = np.asanyarray(color_frame.get_data())
            cv2.imwrite('l515_color_image.jpg', color_image)
            # Perform pose detection
            annotated_frame = pose(image)

            # Display the annotated frame in a cv2 window
            cv2.imshow('Pose Detection', annotated_frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()




def webcam():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Perform pose detection
        annotated_frame = pose(frame)

        # Display the annotated frame in a cv2 window
        #cv2.imwrite('pose_detection.jpg', annotated_frame)
        cv2.imshow('Pose Detection', annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Break the loop when 'q' key is pressed
            
        # Release the webcam and close the cv2 window
    
#webcam()

# import the opencv library
import cv2
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the annotated frame using matplotlib
    plt.imshow(frame)
    plt.pause(0.01)  # Pause briefly to allow the figure to update

    # Check for user input and break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the cv2 window
plt.close()  # Close the matplotlib figure
  
