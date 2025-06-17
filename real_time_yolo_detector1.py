import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet(
    "weights/yolov4-tiny.weights",
    "cfg/yolov4.cfg"
)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video
cap = cv2.VideoCapture("uk.mp4")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

# --- Adjustable Thresholds for Debugging ---
# Set these very low to ensure we capture even faint detections
CONF_THRESHOLD = 0.05 # VERY LOW confidence threshold
NMS_THRESHOLD = 0.1  # VERY LOW NMS threshold (allow many overlapping boxes)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    frame_id += 1
    
    height, width, channels = frame.shape

    # Detecting objects
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
            # No confidence check here yet, we want ALL raw detections
            # if confidence > CONF_THRESHOLD: # This line is commented out for now
            
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates (top-left corner)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # --- Debug Print: Number of raw detections ---
    print(f"Frame {frame_id}: Found {len(boxes)} raw detections.")
    # If len(boxes) > 0, you can also print some raw confidences
    # if len(confidences) > 0:
    #    print(f"  Max raw confidence: {max(confidences):.2f}, Min raw confidence: {min(confidences):.2f}")
    #    print(f"  First 5 raw confidences: {[f'{c:.2f}' for c in confidences[:5]]}")


    # Now apply NMS with very low thresholds
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    # Ensure indexes is not empty and is iterable (flatten if it's a 2D array)
    if len(indexes) > 0:
        if isinstance(indexes[0], np.ndarray):
            indexes = indexes.flatten()
        
        print(f"Frame {frame_id}: Found {len(indexes)} detections after NMS.")

        for i in indexes:
            x, y, w, h = boxes[i]
            
            # Ensure label is valid
            if 0 <= class_ids[i] < len(classes):
                label = str(classes[class_ids[i]])
            else:
                label = "Unknown Class" # Fallback if class_id is out of bounds
            
            confidence = confidences[i]
            color = colors[class_ids[i]] # This assumes class_ids[i] is still valid for colors
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