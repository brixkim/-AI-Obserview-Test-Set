import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI()
model = YOLO("./models/yolov12n-face.onnx")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera open fail")

def gen_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(source=frame, verbose=False)
        annotated = results[0].plot()

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        jpg = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        
@app.get("/")
def video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")