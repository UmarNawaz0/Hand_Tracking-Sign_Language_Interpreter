import cv2
import mediapipe as mp
import time

# Initialize video capture (camera index 0 or 1 based on availability)
cap = cv2.VideoCapture(0)

# Mediapipe Hand Tracking Initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.8,  # Higher confidence for detection
                      min_tracking_confidence=0.8)  # Higher confidence for tracking
mpDraw = mp.solutions.drawing_utils

# For calculating Frames Per Second (FPS)
pTime = 0
cTime = 0

if not cap.isOpened():
    print("Error: Cannot access the camera")
else:
    print("Press 'q' to quit")

while True:
    success, img = cap.read()  # Capture a frame from the camera
    if not success:
        print("Error: Failed to capture frame")
        break

    # Convert the BGR image to RGB for Mediapipe processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # Process the image to detect hands

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # Loop through detected hands
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape  # Get dimensions of the image
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixels
                z = lm.z  # Normalized z-coordinate

                # Print landmark ID and coordinates
                print(f"ID: {id}, X: {cx}, Y: {cy}, Z: {z:.4f}")
# Draw a circle on a specific landmark (e.g., ID 4 - Thumb tip)
                if id == 4:  # Thumb tip
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

            # Draw hand landmarks and connections on the image
            mpDraw.draw_landmarks(
                img, handLms, mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Landmarks
                mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2)  # Connections
            )

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed image
    cv2.imshow("Enhanced Hand Tracking", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
