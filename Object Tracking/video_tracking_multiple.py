# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:08:10 2024

@author: sconc
"""

import sys
import cv2
import numpy as np
import random

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

# Function to choose multiple object categories for tracking
def choose_multiple_categories(detections, confidences, classes, categories, frame):
    print("\nAvailable categories:")
    for i, category in enumerate(categories):
        print(f"{i + 1}. {category}") # print detected categories
    cv2.imshow("frame", frame) # Show the frame
    cv2.waitKey(0)  # Wait indefinitely for a key press

    cv2.destroyAllWindows()

    choices = input("\nChoose the category numbers to track (comma-separated, e.g., 1,3,4): ")
    chosen_indices = [int(choice.strip()) - 1 for choice in choices.split(',')]

    chosen_categories = [categories[choice] for choice in chosen_indices]
    
    chosen_boxes = []
    chosen_confidences = []
    chosen_classIDs = []

    for i, cls in enumerate(classes):
        if CLASSES[cls] in chosen_categories:
            chosen_boxes.append(detections[i])
            chosen_confidences.append(confidences[i])
            chosen_classIDs.append(classes[i])

    return chosen_boxes, chosen_classIDs

# Generate random colors for each object to be tracked
def get_random_colors(num_colors):
    colors = []
    for i in range(num_colors):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return colors

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

# Ask the user to choose multiple categories
chosen_boxes, chosen_classIDs = choose_multiple_categories(boxes, confidences, classIDs, detected_categories, frame)

if not chosen_boxes:
    print("No object found for the chosen categories.")
    sys.exit()

# Generate random colors for the number of objects being tracked
colors = get_random_colors(len(chosen_boxes))

# Initialize trackers for each chosen object
trackers = []
for bbox in chosen_boxes:
    tracker = cv2.TrackerCSRT_create()  # You can replace CSRT with KCF, MIL, etc.
    tracker.init(frame, bbox)
    trackers.append(tracker)

# Define new width and height for resizing the frame (adjust these as needed)
new_width = 640  # for example, resize the frame width to 640 pixels
new_height = int(frame.shape[0] * (new_width / frame.shape[1]))  # maintain aspect ratio

# Loop over the frames of the video
while True:
    # Read a new frame
    ret, frame = cap.read()

    # If the frame was not retrieved, break from the loop
    if not ret:
        break

    # Loop over all trackers
    for i, tracker in enumerate(trackers):
        # Update the tracker
        success, bbox = tracker.update(frame)

        # Draw the bounding box if tracking was successful
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            color = colors[i]  # Assign a unique color for each object
            cv2.rectangle(frame, p1, p2, color, 2, 1)
            # Optionally, display the class name
            class_name = CLASSES[chosen_classIDs[i]]
            cv2.putText(frame, class_name, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, f"Tracking {CLASSES[chosen_classIDs[i]]} failed", (100, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Resize the frame for display
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Display the resized frame
    cv2.imshow("Multi-object Tracking", resized_frame)

    # Exit loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
