# Hand Tracking & Sign Language Interpreter

This project is a **Sign Language Interpreter** using **Machine Learning**, designed specifically for recognizing single-handed alphabets.

## Hand Gesture Recognition using Machine Learning
This project implements a **real-time hand gesture recognition system** using **OpenCV, MediaPipe, and a Random Forest classifier**.

## Features
- **Hand tracking** (`HAND_TRACKING.py`) - Detects and tracks hand landmarks.
- **Image collection** (`collect_imgs.py`) - Captures training images for single-handed alphabets.
- **Dataset creation** (`create_dataset.py`) - Processes collected images and generates dataset.
- **Model training** (`train_classifier.py`) - Trains a classifier for hand gesture recognition.
- **Real-time inference** (`inference_classifier.py`) - Recognizes hand gestures in real-time.

## Usage Guide
1. **Collect Training Data**  
   Run `collect_imgs.py` to capture images for single-handed alphabets.
2. **Create Dataset**  
   Run `create_dataset.py` to generate a pickle file for training.
3. **Train the Model**  
   Run `train_classifier.py` to train the classifier on the dataset.
4. **Perform Real-Time Recognition**  
   Run `inference_classifier.py` to recognize hand gestures.

## Installed Libraries
- **OpenCV (`cv2`)** - Image capture, processing, and display.
- **MediaPipe (`mediapipe`)** - Hand tracking and landmark extraction.
- **NumPy (`numpy`)** - Numerical operations and feature extraction.
- **scikit-learn (`sklearn`)** - Machine learning model training and evaluation.
- **pickle (`pickle`)** - Saving and loading models/datasets.
- **os (`os`)** - File system operations.
- **time (`time`)** - FPS calculations for `HAND_TRACKING.py`.

