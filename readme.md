# YOLO UAV Detection Project

This project utilizes the Ultralytics YOLO (You Only Look Once) model to perform object detection tasks on various media forms including video streams, saved video files, and images. The implementations cover object detection in real-time video from webcams, processed video files, and randomly selected images from a dataset.

## Project Structure

- **Real-Time Webcam Detection**: Uses the YOLO model to detect objects in real-time from a webcam feed.
- **Video File Processing**: Processes a pre-recorded video to detect objects, annotating the frames and saving the output to a new file.
- **Image Detection**: Randomly selects an image from a specified directory and performs object detection, displaying the results.

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Logging

Ensure you have the required hardware support, primarily a compatible GPU for YOLO models if you are utilizing CUDA capabilities for enhanced performance.

## Setup

1. Clone this repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set api key in config.ini (example cant be found in config.sample.ini)
4. Run train.py

## Usage

- For Real-Time Detection: Run the script that accesses the webcam. Ensure you have a functional webcam connected.
- For Video Processing: Provide the path to your video file in the script.
- For Image Detection: Ensure your directory path to the image folder is correct in the script.

## Contribution

Feel free to fork this project and make your own changes too. Pull requests for improvements are welcome.
