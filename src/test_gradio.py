import gradio as gr
import numpy as np
from ultralytics import YOLO

model = YOLO("./models/yolov12n-face.onnx")

def detect(frame):
    if frame is None:
        return None
    results = model.predict(frame, verbose=False)[0]
    annotated = results.plot()
    return annotated

with gr.Blocks() as demo:
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            cam = gr.Image(sources=["webcam"], type="numpy", label="YOLO Test")
            
        cam.stream(fn=detect, inputs=cam, outputs=cam, stream_every = 0.1)

demo.launch(share=True)