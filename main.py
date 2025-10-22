from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')


print(model.task)

# 只检测人类（classes=[0]），置信度阈值大于 0.6
results = model.predict(
    source="./images/test",
    imgsz=640,
    save=True,
    classes=[0],
    conf=0.6
)
