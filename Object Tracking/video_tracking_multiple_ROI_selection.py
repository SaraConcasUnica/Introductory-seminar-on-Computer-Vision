import cv2
import sys

# Path to your video file
video_path = 'esempio.mp4'  # Replace with your video file path

# Initialize the video capture object
video = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    sys.exit()

# Read the first frame of the video
ret, frame = video.read()
if not ret:
    print("Error: Could not read video.")
    exit()

# Resize the first frame to reduce window size
scale_percent = 50  # You can adjust this scale to change the window size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)

# List to store multiple bounding boxes, trackers, and colors
bboxes = []
trackers = []
colors = []

# Define a list of colors for different objects
object_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Select multiple ROIs (bounding boxes)
while True:
    bbox = cv2.selectROI("Select Object", frame, False)
    bboxes.append(bbox)
    
    # Initialize a new tracker for each object
    tracker = cv2.TrackerCSRT_create()  # You can replace CSRT with other algorithms like KCF, MIL, etc.
    tracker.init(frame, bbox)
    trackers.append(tracker)
    
    # Assign a color to each object, cycling through the list of colors
    colors.append(object_colors[len(trackers) % len(object_colors)])
    
    # Ask if the user wants to add another object to track
    print("Press 'n' to select another object or any other key to continue.")
    if cv2.waitKey(0) & 0xFF != ord('n'):
        break

cv2.destroyAllWindows()

# Loop over the frames of the video
while True:
    # Read a new frame
    ret, frame = video.read()
    if not ret:
        break

    # Resize the frame to reduce the window size
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Update all trackers and draw their bounding boxes
    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            # Adjust the bounding box coordinates according to the resized frame
            scaled_bbox = (
                int(bbox[0] * scale_percent / 100), int(bbox[1] * scale_percent / 100),
                int(bbox[2] * scale_percent / 100), int(bbox[3] * scale_percent / 100)
            )
            p1 = (int(scaled_bbox[0]), int(scaled_bbox[1]))
            p2 = (int(scaled_bbox[0] + scaled_bbox[2]), int(scaled_bbox[1] + scaled_bbox[3]))
            cv2.rectangle(resized_frame, p1, p2, colors[i], 2, 1)
        else:
            # Display failure message for that object
            cv2.putText(resized_frame, f"Tracking failed for object {i+1}", (100, 80 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the resized frame
    cv2.imshow("Multi-object Tracking", resized_frame)

    # Exit loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video.release()
cv2.destroyAllWindows()
