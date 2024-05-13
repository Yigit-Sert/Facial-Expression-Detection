# Emotion Detection using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) for real-time emotion detection using facial expressions. The CNN model is trained on the FER2013 dataset, which includes seven different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- Numpy

## Usage

### Training the Model

1. Ensure you have the required dependencies installed.
2. Organize your dataset into training and testing directories (`data/train` and `data/test` respectively).
3. Run `main.py` to train the model. The trained model will be saved as `model_file.h5`.

### Real-time Emotion Detection from Webcam

1. Make sure you have a webcam connected to your system.
2. Run `test.py` to start real-time emotion detection from the webcam feed.
3. Press 'q' to quit the program.

### Emotion Detection from Image

1. Run `testdata.py` to perform emotion detection on a static image (`emot.png`).
2. The detected emotions will be displayed on the image.

## File Descriptions

- `main.py`: Contains the code for model training and evaluation.
- `test.py`: Implements real-time emotion detection using a webcam feed.
- `testdata.py`: Performs emotion detection on a static image.
- `haarcascade_frontalface_default.xml`: Haar cascade file for face detection.
- `model_file.h5`: Trained CNN model.

## Acknowledgments

- The FER2013 dataset for emotion recognition.
- OpenCV for face detection.
- TensorFlow and Keras for deep learning model implementation.
