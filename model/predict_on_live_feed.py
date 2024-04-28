import cv2
import logging
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = "./model/runs/detect/train/weights/best.pt"


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        exit()

    cap.set(3, 640)
    cap.set(4, 480)

    try:
        model = YOLO(MODEL_PATH)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    while True:
        ret, img = cap.read()
        if not ret:
            logging.error("Failed to read from camera.")
            break

        try:
            results = model.predict(img)
            annotator = Annotator(img)

            for r in results:
                for box in r.boxes:
                    annotator.box_label(box.xyxy[0], f'{model.names[int(box.cls)]} {box.conf.cpu().item():.2f}')

            cv2.imshow('YOLOV8 Detection', annotator.result())
            if cv2.waitKey(100) & 0xFF == ord(' '):  # Press space bar to exit
                logging.info("Stopping video stream.")
                break
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()