# Dependencies
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt

# MediaPipe holistic model
mp_holistic = mp.solutions.holistic

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils


def mp_detection(image, model):
    # Color conversion - BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make the image unwriteable to save bandwith
    image.flags.writeable = False

    # Make prediction
    results = model.process(image)

    # Image is now writeable again (next frame)
    image.flags.writeable = True

    # Color conversion - RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def mp_draw_landmarks(image, results):
    # Draw the hand and pose connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 120), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


# Access webcam
capture = cv2.VideoCapture(0)

# Set MediaPipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # While the webcam is open
    while capture.isOpened():

        # Read the webcam feed
        ret, frame = capture.read()

        # Make detections
        image, results = mp_detection(frame, holistic)

        # Draw landmarks
        mp_draw_landmarks(image, results)

        # Show the webcam feed to the window
        cv2.imshow("OpenCV Video Feed", image)

        # Break the loop when q is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Stop the webcam feed
    capture.release()

    # Close the webcam window
    cv2.destroyAllWindows()


def extract_keypoints(results):
    # If there is data collected for the face, hands, and pose; extract the keypoint values
    if results.face_landmarks:
        face = np.array([[i.x, i.y, i.z] for i in results.pose_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[i.x, i.y, i.z] for i in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    if results.left_hand_landmarks:
        lh = np.array([[i.x, i.y, i.z] for i in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    return np.concatenate([face, rh, lh, pose])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join("MP_Data")

# Signs to detect
actions = np.array(["hello", "thanks", "iloveyou"])

#
no_sequences = 30

# Videos are 30 frames in length
sequence_length = 30