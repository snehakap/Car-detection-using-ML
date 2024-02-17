import cv2
from tracker import EuclideanDistTracker
from google.colab.patches import cv2_imshow
import os

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("/content/input_video.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100)

# Create a directory to save frames
output_directory = "/content/frames"
os.makedirs(output_directory, exist_ok=True)

frame_count = 0
total_cars_passed = -1
prev_objects = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[190:352, 270:600]

    # Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 7000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])

    # Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if y < 10 and id not in prev_objects:  # Check if the object crosses the top boundary
            total_cars_passed += 1
            prev_objects.append(id)

    # Display the count on top of the frame
    cv2.putText(frame, f"Total Cars Passed: {total_cars_passed}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    #cv2_imshow(frame)

    # Save the frame
    frame_count += 1
    frame_filename = os.path.join(output_directory, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

#Convert the frames to video format
!ffmpeg -i /content/frames/frame_%04d.jpg -c:v libx264 -vf "fps=30,format=yuv420p" /content/output_video.mp4
