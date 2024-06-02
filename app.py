from flask import Flask, render_template, Response
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time

app = Flask(__name__)

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(0)
tracker = Tracker()

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

cy1 = 322
cy2 = 368
offset = 6
vh_down = {}
counter = []
vh_up = {}
counter1 = []

def process_frame():
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'cars' in c or 'truck' in c or 'bus' in c or 'motorcycles' in c:
                list.append([x1, y1, x2, y2])
        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            if cy1 < (cy + offset) and cy1 > (cy - offset):
                vh_down[id] = time.time()
            if id in vh_down:
                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    elapsed_time = time.time() - vh_down[id]
                    if counter.count(id) == 0:
                        counter.append(id)
                        distance = 10  # meters
                        a_speed_ms = distance / elapsed_time
                        a_speed_kh = a_speed_ms * 3.6
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            if cy2 < (cy + offset) and cy2 > (cy - offset):
                vh_up[id] = time.time()
            if id in vh_up:
                if cy1 < (cy + offset) and cy1 > (cy - offset):
                    elapsed1_time = time.time() - vh_up[id]
                    if counter1.count(id) == 0:
                        counter1.append(id)
                        distance1 = 25  # meters
                        a_speed_ms1 = distance1 / elapsed1_time
                        a_speed_kh1 = a_speed_ms1 * 3.6
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        cv2.line(frame, (150, cy1), (927, cy1), (255, 255, 255), 1)
        cv2.putText(frame, ('L1'), (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(frame, (150, cy2), (927, cy2), (255, 255, 255), 1)
        cv2.putText(frame, ('L2'), (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        d = len(counter)
        u = len(counter1)
        # cv2.putText(frame, ('goingdown:-') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        # cv2.putText(frame, ('goingup:-') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
