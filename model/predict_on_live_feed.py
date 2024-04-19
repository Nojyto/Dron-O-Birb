import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


MODEL_PATH = "./model/runs/detect/train4/weights/best.pt"


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    model = YOLO(MODEL_PATH)

    while True:
        _, img = cap.read()
        results = model.predict(img)
        annotator = Annotator(img)

        for r in results:
            for box in r.boxes:
                annotator.box_label(box.xyxy[0], f'{model.names[int(box.cls)]} {box.conf.cpu().item():.2f}')
        
        cv2.imshow('YOLOV8 Detection', annotator.result())     
        if cv2.waitKey(100) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()