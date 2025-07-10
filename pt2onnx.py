import torch

from ultralytics import YOLO

model = YOLO(r'D:\work\ultralytics-main\runs\obb\train32\weights\best.pt')
model.export(format='onnx',opset=11)
