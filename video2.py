import cv2
import mediapipe as mp
import math
import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = False  # prevent cursor going to top-left corner to stop script

# Webcam
cap = cv2.VideoCapture(0)

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# Eye & blink globals
eye_closed = False
eye_closed_prev = False
blink_count = 0

# Smoothing globals
smoothed_x, smoothed_y = 0.5, 0.5
alpha = 0.2  # smoothing factor

# Screen size
screen_width, screen_height = pyautogui.size()


def map_range(value, old_min, old_max, new_min=0, new_max=1):
    # Clamp value to old range
    value = max(min(value, old_max), old_min)
    # Map to new range
    return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def getIrisPos(frame):
    global smoothed_x, smoothed_y

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            # Landmarks
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]

            left_corner = face_landmarks.landmark[33]
            right_corner = face_landmarks.landmark[133]
            top = face_landmarks.landmark[159]
            bottom = face_landmarks.landmark[145]

            # Pixel coords
            lx, ly = int(left_corner.x * iw), int(left_corner.y * ih)
            rx, ry = int(right_corner.x * iw), int(right_corner.y * ih)
            tx, ty = int(top.x * iw), int(top.y * ih)
            bx, by = int(bottom.x * iw), int(bottom.y * ih)
            ix, iy = int(left_iris.x * iw), int(left_iris.y * ih)

            eye_width = rx - lx
            eye_height = by - ty

            rel_x = (ix - lx) / eye_width
            rel_y = (iy - ty) / eye_height

            # Mirror for webcam display
            rel_x = rel_x

            # Map iris range to 0-1 (calibrate these ranges for your eyes)
            rel_x = map_range(rel_x, 0.5, 0.55)
            rel_y = map_range(rel_y, 0.35, 0.5)
            print(rel_y)

            # Clamp 0-1
            rel_x = max(0, min(rel_x, 1))
            rel_y = max(0, min(rel_y, 1))

            # Smooth movement
            smoothed_x = smoothed_x * (1 - alpha) + rel_x * alpha
            smoothed_y = smoothed_y * (1 - alpha) + rel_y * alpha

            mouse_x = smoothed_x * screen_width
            mouse_y = smoothed_y * screen_height
            pyautogui.moveTo(mouse_x, mouse_y)

            # Draw iris
            cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1)
            cv2.circle(
                frame,
                (int(right_iris.x * iw), int(right_iris.y * ih)),
                3,
                (0, 255, 0),
                -1,
            )


def detectEyes(frame):
    global eye_closed, eye_closed_prev, blink_count

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_indices = [33, 133, 159, 145, 33]
            right_eye_indices = [362, 263, 386, 374, 362]

            left_eye = [
                (
                    int(face_landmarks.landmark[i].x * frame.shape[1]),
                    int(face_landmarks.landmark[i].y * frame.shape[0]),
                )
                for i in left_eye_indices
            ]
            right_eye = [
                (
                    int(face_landmarks.landmark[i].x * frame.shape[1]),
                    int(face_landmarks.landmark[i].y * frame.shape[0]),
                )
                for i in right_eye_indices
            ]

            top_left = face_landmarks.landmark[159]
            bottom_left = face_landmarks.landmark[145]
            left_corner = face_landmarks.landmark[33]
            right_corner = face_landmarks.landmark[133]

            ratio = math.sqrt(
                (top_left.x - bottom_left.x) ** 2 + (top_left.y - bottom_left.y) ** 2
            ) / math.sqrt(
                (left_corner.x - right_corner.x) ** 2
                + (left_corner.y - right_corner.y) ** 2
            )

            for x, y in left_eye + right_eye:
                color = (0, 0, 255) if ratio < 0.4 else (0, 255, 0)
                cv2.circle(frame, (x, y), 2, color, -1)

            if eye_closed_prev and not (ratio < 0.4):
                blink_count += 1
                # pyautqogui.click()

            eye_closed = ratio < 0.4

    eye_closed_prev = eye_closed


cv2.namedWindow("Data")
data_img = np.zeros((200, 400, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    detectEyes(frame)
    getIrisPos(frame)

    # Draw face boxes
    for x, y, w, h in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0) if w * h <= 250000 else (0, 0, 255),
            2,
        )

    # Show windows
    cv2.imshow("Faces Detected", frame)

    data_img[:] = 0
    cv2.putText(
        data_img,
        f"Blinks: {blink_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Data", data_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
