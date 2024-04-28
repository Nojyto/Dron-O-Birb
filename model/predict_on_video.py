import cv2
import os
import logging
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VIDEO_PATH = '../datasets/test_grass.mp4'
VIDEO_PATH_OUT = os.path.join(os.path.dirname(VIDEO_PATH), f"{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_out{os.path.splitext(VIDEO_PATH)[1]}")

MODEL_PATH = "./runs/detect/train/weights/best.pt"
THRESHOLD = 0.1


if __name__ == '__main__':
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            logging.error("Error opening video file.")
            raise ValueError("Unable to open video file.")
        
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read the first frame of the video.")
            raise ValueError("Unable to read the first frame of the video.")

        H, W, _ = frame.shape
        out = cv2.VideoWriter(VIDEO_PATH_OUT, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        model = YOLO(MODEL_PATH)
        logging.info("Model loaded successfully.")

        while ret:
            results = model.predict(frame)
            annotator = Annotator(frame)
            
            for r in results:
                for box in r.boxes:
                    if box.conf > THRESHOLD:
                        annotator.box_label(box.xyxy[0], f'{model.names[int(box.cls)]} {box.conf.cpu().item():.2f}')

            out.write(annotator.result())
            ret, frame = cap.read()
            if not ret and cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                logging.warning("Unexpected end of video.")

        logging.info("Video processing completed. Output saved to: %s", VIDEO_PATH_OUT)
    except Exception as e:
        logging.exception("An error occurred: %s", str(e))
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()