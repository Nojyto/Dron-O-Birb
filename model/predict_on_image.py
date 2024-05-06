import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from helpers import select_random_image


IMAGE_FOLDER_PATH = '../datasets/UAVDetection/valid/images/'
MODEL_PATH = "./runs/detect/train/weights/best.pt"


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