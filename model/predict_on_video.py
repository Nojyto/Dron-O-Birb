import cv2
import os
import logging
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


VIDEO_PATH = './datasets/test2.mp4'
VIDEO_PATH_OUT = os.path.join(os.path.dirname(VIDEO_PATH), f"{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_out{os.path.splitext(VIDEO_PATH)[1]}")

MODEL_PATH = "./model/runs/detect/train4/weights/best.pt"
THRESHOLD = 0.2


if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(VIDEO_PATH_OUT, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model = YOLO(MODEL_PATH)
    logging.getLogger('ultralytics').setLevel(logging.ERROR)

    while ret:
        results = model.predict(frame)
        annotator = Annotator(frame)
        
        for r in results:
            for box in r.boxes:
                if box.conf > THRESHOLD:
                    annotator.box_label(box.xyxy[0], f'{model.names[int(box.cls)]} {box.conf.cpu().item():.2f}')

        out.write(annotator.result())
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed. Output saved to:", VIDEO_PATH_OUT)