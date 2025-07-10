from ultralytics import YOLO


def main():
    model = YOLO('yolov8-obb.yaml').load('yolov8s-obb.pt')  # build from YAML and transfer weights
    model.train(data='dota8-obb.yaml', epochs=300, imgsz=1024, batch=4, workers=4)


if __name__ == '__main__':
    main()