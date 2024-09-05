# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:43:08 2024

@author: sconc
"""

import cv2
import sys

# Path to your video file
video_path = 'corsa.mp4'  # Replace with your video file path

# Initialize the video capture object
video = cv2.VideoCapture(video_path) #  This function is used to capture video from a file or a camera. 

# Check if the video opened successfully
if not video.isOpened(): #  checks if the video file was successfully opened. I
    print("Error: Could not open video.")
    sys.exit() # if not, an error message is printed, and the program exits.

# Read the first frame of the video
ret, frame = video.read() # Reads the first frame from the video
if not ret: #  ret is a boolean that indicates if the frame was successfully read
    print("Error: Could not read video.")
    exit()

# Manually select the bounding box of the object to track (ROI)
bbox = cv2.selectROI(frame, False) #  Opens a window where you can manually select the Region of Interest (ROI) by drawing a bounding box around the object you want to track.
cv2.destroyAllWindows()

# Initialize the tracker (You can choose among several available algorithms)
tracker = cv2.TrackerCSRT_create()  # You can replace CSRT with KCF, MIL, etc.
# Creates a CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) tracker object. 
# Initialize the tracker with the first frame and the bounding box
tracker.init(frame, bbox)

# Loop over the frames of the video
while True:
    # Read a new frame
    ret, frame = video.read()

    # If the frame was not retrieved, break from the loop
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)
    #  success (a boolean indicating if tracking was successful)
    # bbox (the updated bounding box)
    
    # Draw the bounding box if tracking was successful
    if success:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame) 

    # Exit loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()
cv2.destroyAllWindows()
