import cv2
import numpy as np
import time

# --- Configuration for Detection ---
CONF_THRESHOLD = 0.5 # Minimum confidence to consider a detection. Adjust as needed.
NMS_THRESHOLD = 0.4  # IoU threshold for Non-Maximum Suppression. Adjust as needed.

# Load Yolo
# CRITICAL FIX: Ensure yolov4-tiny.cfg matches yolov4-tiny.weights
net = cv2.dnn.readNet("weights/yolov4-tiny.weights", "cfg/yolov4.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

# FIX: Corrected Index Error: i[0] - 1 to i - 1
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- Webcam Integration ---
# 0 usually refers to the default webcam. If you have multiple, try 1, 2, etc.
cap = cv2.VideoCapture(0) 

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    ret, frame = cap.read() # Read frame from webcam
    if not ret:
        print("Failed to grab frame from webcam. Exiting...")
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

        for i in indexes:
            x, y, w, h = boxes[i]
            
            # Ensure class_id is within bounds of classes list
            if 0 <= class_ids[i] < len(classes):
                label = str(classes[class_ids[i]])
            else:
                label = "Unknown" # Fallback if class_id is out of bounds
            
            confidence = confidences[i]
            # FIX: Use class_ids[i] for color, not just 'i'
            color = colors[class_ids[i]] 
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow("Image", frame) # Display the webcam feed with detections
    
    key = cv2.waitKey(1)
    if key == 27: # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()