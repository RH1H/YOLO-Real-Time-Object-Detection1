import cv2
import numpy as np
import time

# --- Configuration for Detection ---
CONF_THRESHOLD = 0.2 # Minimum confidence to consider a detection. Adjust as needed (e.g., 0.1 for more detections).
NMS_THRESHOLD = 0.3  # IoU threshold for Non-Maximum Suppression. Adjust as needed (e.g., 0.2 to allow more overlapping boxes).

# Load Yolo
# CRITICAL FIX: The .cfg file MUST match the .weights file exactly.
# Changed "cfg/yolov4.cfg" to "cfg/yolov4-tiny.cfg"
net = cv2.dnn.readNet("weights/yolov4-tiny.weights", "cfg/yolov4.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

# FIX: Corrected Index Error: i[0] - 1 to i - 1
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video
# Ensure 'usa-street.mp4' is in the same directory as your script, or provide its full path
cap = cv2.VideoCapture("usa-street.mp4")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
frames_with_detections = 0 # Initialize counter for frames with detections

while True:
    ret, frame = cap.read() # Read frame and check if successful
    if not ret: # Break loop if no more frames (end of video)
        print("End of video or failed to read frame.")
        break

    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    # Input size for YOLOv4-tiny is typically 416x416
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONF_THRESHOLD: # Use the adjustable confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                # FIX: detection[2] is width, detection[3] is height in YOLO output
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # FIX: Rectangle coordinates (top-left corner) calculation
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression (NMS) with adjustable thresholds
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    # Ensure indexes is not empty and is iterable (flatten if it's a 2D array)
    if len(indexes) > 0:
        if isinstance(indexes[0], np.ndarray): # Check if elements are arrays (common in some OpenCV versions)
            indexes = indexes.flatten() # Flatten to a 1D array
        
        # Increment counter if detections are found in this frame
        frames_with_detections += 1 

        for i in indexes:
            x, y, w, h = boxes[i]
            
            # Ensure class_id is within bounds of classes list
            if 0 <= class_ids[i] < len(classes):
                label = str(classes[class_ids[i]])
            else:
                label = "Unknown" # Fallback if class_id is out of bounds
            
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    
    key = cv2.waitKey(1)
    if key == 27: # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()

# Print the total number of frames with detections after the loop finishes
print(f"\nTotal frames processed: {frame_id}")
print(f"Total frames with at least one object detected: {frames_with_detections}")