import cv2
import os
import random
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


IMAGE_FOLDER_PATH = './datasets/VisDrone/VisDrone2019-DET-test-dev/images'
MODEL_PATH = "./model/runs/detect/train4/weights/best.pt"


def select_random_image(folder_path: str) -> str | None:
    files = os.listdir(folder_path)
    images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        random_image = random.choice(images)
        print(f"Selected random image: {random_image}")
        return os.path.join(folder_path, random_image)
    else:
        print("No images found in the directory.")
        return None


def detect_objects_in_image(image_path: str) -> None:
    model = YOLO(MODEL_PATH)
    results = model.predict(image_path)
    
    img = cv2.imread(image_path)
    annotator = Annotator(img)
    
    for r in results:
        for box in r.boxes:
            annotator.box_label(box.xyxy[0], f'{model.names[int(box.cls)]} {box.conf.cpu().item():.2f}')
        
    cv2.imshow('Detected Image', annotator.result())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = select_random_image(IMAGE_FOLDER_PATH)
    detect_objects_in_image(image_path)