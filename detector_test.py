from ultralytics import YOLO


model = YOLO('yolov10x.pt')

results = model(
    conf=0.1, iou=0.5, imgsz=1280, source='Sample.mp4',
    save=True, show=False, classes=0)
