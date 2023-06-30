from ultralytics import YOLO

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['bus_copy.jpg'])  # return a list of Results objects

print(results)