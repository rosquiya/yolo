import torch
import torch.onnx
from ultralytics import YOLO

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m-cls.pt')  # load an official model
model = YOLO('./runs/classify/train/weights/best.pt') # load a custom trained model

# Export the model
model.export(format="onnx")