import time
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from predict_on_image import MODEL_PATH, select_random_image

WINDOW_W = 640
WINDOW_H = WINDOW_W
STEP_SIZE = int(WINDOW_W / 1.5)


# https://github.com/ultralytics/ultralytics/issues/5325
def extend_window_with_solid_edges(source_image):
    extended_image = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    extended_image[0:source_image.shape[0], 0:source_image.shape[1]] = source_image
    return extended_image


def generate_sliding_window(image, stepSize, windowSize):
    for y in range(0, (image.shape[0] - WINDOW_H + stepSize), stepSize):
        for x in range(0, (image.shape[1] - WINDOW_W + stepSize), stepSize):
            window = image[y:(y + windowSize[0]), x:(x + windowSize[1])]
            if window.shape[1] < WINDOW_W or window.shape[0] < WINDOW_H:
                window = extend_window_with_solid_edges(window)
            yield x, y, window


IMAGE_FOLDER_PATH = "../datasets/Test/DJI/M3/building_side"
TEST_IMAGE_PATH = f"{IMAGE_FOLDER_PATH}/DJI_20240421142441_0016_D.JPG"
IMAGE_OUTPUT_PATH = "../testout"

if __name__ == '__main__':
    image_path = TEST_IMAGE_PATH
    # image_path = select_random_image(IMAGE_FOLDER_PATH)

    # Prepare sliding window
    image = cv2.imread(image_path)
    sliding_window = generate_sliding_window(
        image,
        stepSize=STEP_SIZE,
        windowSize=(WINDOW_H, WINDOW_W)
    )

    # Inference
    result_image = image.copy() # Full unsliced image with all the detections
    model = YOLO(MODEL_PATH)

    for (x, y, window) in sliding_window:
        results = model.predict(window)
        for r in results:
            for box in r.boxes:
                xyxy = [int(t) for t in box.xyxy[0].numpy()]  # Converts tensor to normal array
                cv2.rectangle(result_image, (xyxy[0] + x, xyxy[1] + y), (xyxy[2] + x, xyxy[3] + y), (0, 0, 255), 2)
                cv2.rectangle(window, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            # if len(r.boxes) > 0:
            #     cv2.imshow("Result", window)
            #     cv2.waitKey(200)

        # cv2.imwrite(f"{IMAGE_OUTPUT_PATH}/{y}_{x}.jpg", window) #Write all windows to disk
        cv2.imshow("Result", window)
        cv2.waitKey(1)

    aspect_ratio = image.shape[0] / image.shape[1]
    result_image = cv2.resize(result_image, (1800, int(1800 * aspect_ratio)))  # Resize for viewing
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
