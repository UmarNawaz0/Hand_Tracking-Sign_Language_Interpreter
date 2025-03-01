import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error. Exiting...")
            break
        cv2.putText(frame, 'Ready? Press "Q" to start.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for counter in range(dataset_size):
        ret, frame = cap.read()
        if not ret:
            print("Camera error. Exiting...")
            break
        resized_frame = cv2.resize(frame, (224, 224))  # Ensure uniform image size
        cv2.imshow('frame', resized_frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), resized_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing Esc
            break

cap.release()
cv2.destroyAllWindows()
