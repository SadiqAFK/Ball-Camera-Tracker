from ultralytics import YOLO
import torch
import os

model = YOLO('yolov8n.pt')

if __name__ == '__main__':      
    results = model.train(data="config.yaml", epochs=50, patience=5)