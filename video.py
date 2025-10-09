import cv2
import mediapipe as mp
import math
import pyautogui
import time
import numpy as np

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils
eye_closed = False
blink_count = 0
eye_closed_prev = False
faceArea = 0
distance = 0

# def getIrisPos(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             ih, iw, _ = frame.shape
#             left_iris = face_landmarks.landmark[474]
#             right_iris = face_landmarks.landmark[469]

#             left_x, left_y = int(left_iris.x * iw), int(left_iris.y * ih)
#             right_x, right_y = int(right_iris.x * iw), int(right_iris.y * ih)

#             left_corner = face_landmarks.landmark[33]
#             right_corner = face_landmarks.landmark[133]
#             top = face_landmarks.landmark[159]
#             bottom = face_landmarks.landmark[145]
#             iris = face_landmarks.landmark[474]  # left iris

#             # convert normalized coordinates to image coords
#             iw, ih = frame.shape[1], frame.shape[0]
#             lx, ly = int(left_corner.x * iw), int(left_corner.y * ih)
#             rx, ry = int(right_corner.x * iw), int(right_corner.y * ih)
#             tx, ty = int(top.x * iw), int(top.y * ih)
#             bx, by = int(bottom.x * iw), int(bottom.y * ih)
#             ix, iy = int(iris.x * iw), int(iris.y * ih)

#             # width and height of eye box
#             eye_width = rx - lx
#             eye_height = by - ty

#             # relative iris position (0.0 = left/top, 1.0 = right/bottom)
#             rel_x = (ix - lx) / eye_width
#             rel_y = (iy - ty) / eye_height

#             screen_width, screen_height = pyautogui.size()

#             mouse_x = rel_x * screen_width
#             mouse_y = rel_y * screen_height

#             pyautogui.moveTo(mouse_x, mouse_y)

#             cv2.circle(frame, (left_x, left_y), 3, (0, 255, 0), -1)
#             cv2.circle(frame, (right_x, right_y), 3, (0, 255, 0), -1)


def detectEyes(frame):
    global blink_count
    global eye_closed
    global eye_closed_prev
    global right_corner
    global top_left
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_indices = [33, 133, 159, 145, 133]  # example
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
                math.pow(top_left.x - bottom_left.x, 2)
                + math.pow(top_left.y - bottom_left.y, 2)
            ) / math.sqrt(
                math.pow(left_corner.x - right_corner.x, 2)
                + math.pow(left_corner.y - right_corner.y, 2)
            )

            for x, y in left_eye + right_eye:
                if ratio < 0.4:
                    eye_closed = True
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                else:
                    eye_closed = False
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            if eye_closed_prev and not eye_closed:
                blink_count += 1

    eye_closed_prev = eye_closed


def detectFace(frame):
    global faceArea
    global distance
    if len(faces) > 0:
        x, y, w, h = faces[0]  # first detected face only
        faceArea = w * h
        distance = (150000 * 2) / (w * h) * 12

        cv2.putText(
            frame,
            "W: " + str(w) + " H:" + str(h),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if w * h > 180000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


start_time = time.time()


def putTextOnImage(image, text, ypos):
    cv2.putText(
        image,
        text,
        (10, ypos),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )


cv2.namedWindow("Faces Detected")
cv2.namedWindow("Data")
averageDistance = 0
distanceMeasures = 0


def drawData():
    global averageDistance
    global distanceMeasures
    data_img[:] = 0  # clear previous
    putTextOnImage(data_img, f"Blinks: {blink_count}", 50)
    elapsed = time.time() - start_time

    putTextOnImage(
        data_img, f"Blink rate: {round((blink_count * 3) / elapsed * 100)}%", 100
    )
    putTextOnImage(data_img, f"Face Distance: {distance:.2f}in", 150)
    if distance < 500:
        averageDistance += distance
        distanceMeasures += 1
    putTextOnImage(
        data_img,
        f"Avg Face Distance: {averageDistance / distanceMeasures:.2f}in",
        200,
    )

    # putTextOnImage(data_img, f"Time: {elapsed:.2f}", 250)


data_img = np.zeros((300, 600, 3), dtype=np.uint8)
last_time = time.time()
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Failed to grab frame")
        break
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    detectFace(frame)
    detectEyes(frame)
    if time.time() - last_time > 1:
        drawData()
        last_time = time.time()

    cv2.imshow("Faces Detected", frame)
    cv2.imshow("Data", data_img)
    # Show data window

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
