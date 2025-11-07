import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("./models/yolov12n-face.onnx")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open fail")
    exit()

while True:
    status, img = cap.read()

    if not status:
        print("Can't read Camera")
        break

    results = model.predict(source=img)
    plots = results[0].plot()
    cv2.imshow("Camera", plots)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()