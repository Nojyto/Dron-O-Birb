import torch
import platform
from ultralytics import YOLO

DATA_PATH = "../datasets/config.yaml"
BASE_MODEL = "yolov8n.pt" # pt for pretrained yaml for new model

def main():
    # Choose the device based on availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the model
    try:
        model = YOLO(BASE_MODEL).to(device)
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

    # Train the model
    try:
        results = model.train(data=DATA_PATH, epochs=1, imgsz=640, device=device)
    except Exception as e:
        raise Exception(f"Error training model: {e}")


if __name__ == '__main__':
    if platform.system() == 'Windows':
        from multiprocessing import freeze_support
        freeze_support()
    main()