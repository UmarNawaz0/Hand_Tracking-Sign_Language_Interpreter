# Hand_Tracking & Sign_Language_Interpreter
It's Sign Language Interpreter using Machine and Deep Learning, but here we're only making this project for Single Handed alphabets

# Hand Gesture Recognition using Machine Learning
This project implements a real-time hand gesture recognition system using OpenCV, MediaPipe, and a Random Forest classifier.

## Features
- Hand tracking with `HAND_TRACKING.py`.
- Image collection for training with `collect_imgs.py`.
- Dataset creation with `create_dataset.py`.
- Model training using `train_classifier.py`.
- Real-time gesture inference with `inference_classifier.py`.

The easiest way to make this project work is by running each file at single time. 
First Run the Collect Images file for collecting entire data for Alphabets only for single hand, then run the create dataset file so it can make a pickle file, after that run the train classifier file so it can train our entire dataset for recognizing. And Lastly run the interference file for recognizing the alphabets for single handed. 


### **Installed Libraries:**
1. **OpenCV (`cv2`)** - Used for image capture, processing, and displaying results.
2. **MediaPipe (`mediapipe`)** - Used for hand tracking and extracting hand landmarks.
3. **NumPy (`numpy`)** - Used for handling numerical computations and feature extraction.
4. **scikit-learn (`sklearn`)** - Used for training and evaluating the machine learning model.
5. **pickle (`pickle`)** - Used for saving and loading the trained model and dataset.
6. **os (`os`)** - Used for handling file system operations.
7. **time (`time`)** - Used for calculating FPS in the `HAND_TRACKING.py` script.

