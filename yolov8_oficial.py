import torch
import multiprocessing
from ultralytics import YOLO




def main():
    # Optional: Set multiprocessing start method (recommended for PyTorch on Windows)
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load pretrained YOLO model
    model = YOLO('yolov8m-cls.pt') #cambiar variador 
    
    # Train the model
    model.train(data=r'C:\Users\Rosario\Videos\ultimo_aliento\tesis_2\data\data_ros_aug', epochs=100, imgsz=320, batch=16,patience=15)

if __name__ == '__main__':
    main()


