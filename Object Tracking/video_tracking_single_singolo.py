# -*- coding: utf-8 -*-
"""
Adapted for MobileNet SSD
"""

import sys
import cv2
import numpy as np
import time

# Load the pre-trained MobileNet SSD model using Caffe framework
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt',
                               'MobileNetSSD_deploy.caffemodel')

# Load the COCO class names
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Function to run detection on a single frame using MobileNet SSD
def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []
    classIDs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])  # Class index
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            boxes.append((startX, startY, endX - startX, endY - startY))
            confidences.append(float(confidence))
            classIDs.append(idx)

    return boxes, confidences, classIDs

# Function to choose the object category for tracking
def choose_category(detections, confidences, classes, categories, frame):
    print("\nAvailable categories:")
    for i, category in enumerate(categories):
        print(f"{i + 1}. {category}")
    cv2.imshow("frame", frame)
    cv2.waitKey(0)  # Waits indefinitely for a key press

    cv2.destroyAllWindows()

    choice = int(input("\nChoose the category number to track: ")) - 1
    chosen_category = categories[choice]

    best_idx = None
    best_score = -1

    for i, cls in enumerate(classes):
        if CLASSES[cls] == chosen_category and confidences[i] > best_score:
            best_idx = i
            best_score = confidences[i]

    return best_idx

# Path to the video file
video_path = 'esempio.mp4'  # Replace with your video file path

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

# Read the first frame from the video
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video.")
    sys.exit()


# Detect objects in the first frame
boxes, confidences, classIDs = detect_objects(frame)

# Extract detected categories and convert to a set for unique values
detected_categories = list(set([CLASSES[cls] for cls in classIDs]))

# Ask the user to choose a category
chosen_idx = choose_category(boxes, confidences, classIDs, detected_categories, frame)

if chosen_idx is None:
    print("No object found for the chosen category.")
    sys.exit()

# Get the bounding box for the chosen object
bbox = boxes[chosen_idx]


# Initialize the tracker (You can choose among several available algorithms)
tracker = cv2.TrackerCSRT_create()  # You can replace CSRT with KCF, MIL, etc.
# Creates a CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) tracker object. 
# Initialize the tracker with the first frame and the bounding box
tracker.init(frame, bbox)

# Loop over the frames of the video
while True:
    # Read a new frame
    ret, frame = cap.read()

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
cap.release()
cv2.destroyAllWindows()
