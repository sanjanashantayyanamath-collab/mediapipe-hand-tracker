

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download the hand landmarker model if needed
import urllib.request
import os

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand detection model...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded successfully!")

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Hand Detection started. Press 'q' to quit.")
print("Detecting hands from webcam...")

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def draw_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape
    
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    return frame

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            frame = draw_landmarks(frame, hand_landmarks)
        
        num_hands = len(detection_result.hand_landmarks)
        cv2.putText(frame, f"Hands detected: {num_hands}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hands detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Hand Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Hand detection ended.")
