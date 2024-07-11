import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Function to get RGB values on mouse movement
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load video
cap = cv2.VideoCapture('2.mp4')

# Retrieve frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame Rate: {fps}")

# Load class labels
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize tracker
tracker = Tracker()

# Line positions and offset for detection
cy1 = 322
cy2 = 368
offset = 6

vh_down = {}
counter = []
vh_up = {}
counter1 = []

# Frame counter
count = 0

while True:    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    
    count += 1

    results = model.predict(frame)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    detections = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, conf, cls = row
        x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)
        class_name = class_list[cls]

        # Filter for vehicles only
        if class_name in ['car', 'truck', 'bus', 'motorcycle']:
            detections.append([x1, y1, x2, y2])

    bbox_id = tracker.update(detections)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if cy1 - offset < cy < cy1 + offset:
            vh_down[id] = count / fps
        if id in vh_down and cy2 - offset < cy < cy2 + offset:
            elapsed_time = (count / fps) - vh_down[id]
            if id not in counter:
                counter.append(id)
                distance = 10  # meters
                a_speed_ms = distance / elapsed_time
                a_speed_kh = a_speed_ms * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"{int(a_speed_kh)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if cy2 - offset < cy < cy2 + offset:
            vh_up[id] = count / fps
        if id in vh_up and cy1 - offset < cy < cy1 + offset:
            elapsed1_time = (count / fps) - vh_up[id]
            if id not in counter1:
                counter1.append(id)
                distance1 = 30  # meters
                a_speed_ms1 = distance1 / elapsed1_time
                a_speed_kh1 = a_speed_ms1 * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"{int(a_speed_kh1)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (150, cy1), (927, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (30, 450), (1000, 450), (255, 255, 255), 1)
    cv2.putText(frame, 'L2', (45, 422), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    d = len(counter)
    u = len(counter1)
    cv2.putText(frame, f'going up: {u}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
