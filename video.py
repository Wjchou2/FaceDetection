import cv2
import mediapipe as mp
import math
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
tilt = 0
landmarks = ""


def get_expression(frame, face_landmarks):
    ih, iw, _ = frame.shape
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]

    lx, ly = int(left_mouth.x * iw), int(left_mouth.y * ih)
    rx, ry = int(right_mouth.x * iw), int(right_mouth.y * ih)
    tx, ty = int(top_lip.x * iw), int(top_lip.y * ih)
    bx, by = int(bottom_lip.x * iw), int(bottom_lip.y * ih)

    cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
    cv2.circle(frame, (rx, ry), 2, (0, 255, 0), -1)
    cv2.circle(frame, (tx, ty), 2, (0, 0, 255), -1)
    cv2.circle(frame, (bx, by), 2, (0, 0, 255), -1)

    if ly + 15 < ty:
        return "Smiling"
    else:
        return "Neutral"


def detectEyes(frame):
    global blink_count
    global eye_closed
    global eye_closed_prev
    global right_corner
    global top_left
    global tilt
    global landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks
            get_expression(frame, face_landmarks)

            left_eye_indices = [33, 133, 159, 145, 133]
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
            top_right = face_landmarks.landmark[386]
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
            x1 = top_right.x
            x2 = top_left.x
            y1 = top_right.y
            y2 = top_left.y
            tilt = math.atan2(abs(y2 - y1), abs(x2 - x1))
            tilt = round((-1 if y2 > y1 else 1) * math.degrees(tilt))

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
        x, y, w, h = faces[0]
        faceArea = w * h
        distance = (150000 * 2) / (w * h) * 12
        if distance < 500:
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


def putTextOnImage(image, text, ypos, color):
    cv2.putText(
        image,
        text,
        (10, ypos),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255 if color else 0, 0 if color else 255),
        2,
    )


cv2.namedWindow("Camera")
cv2.namedWindow("Data")
averageDistance = 0
distanceMeasures = 1
averageTilt = 0
tiltMeasures = 0
smileCount = 0
frameCount = 0


def drawData():
    global averageDistance
    global distanceMeasures
    global smileCount
    global frameCount
    global averageTilt
    global tiltMeasures
    global tilt
    global frame
    data_img[:] = 0
    putTextOnImage(data_img, f"Blinks: {blink_count}", 50, True)
    elapsed = time.time() - start_time
    rate = round((blink_count * 3) / elapsed * 100)
    putTextOnImage(data_img, f"Blink rate: {rate}%", 100, rate > 80)
    dist = round(distance)
    putTextOnImage(data_img, f"Face Distance: {dist}in", 150, dist > 20)
    if distance < 500:
        averageDistance += distance
        distanceMeasures += 1
    avgDist = round(averageDistance / distanceMeasures)
    putTextOnImage(data_img, f"Avg Face Distance: {avgDist}in", 200, avgDist > 20)

    putTextOnImage(data_img, f"Head Tilt: {tilt}deg", 250, abs(tilt) < 15)
    averageTilt += tilt
    tiltMeasures += 1
    avgTilt = round(averageTilt / tiltMeasures)
    putTextOnImage(data_img, f"Avg Head Tilt: {avgTilt}deg", 300, abs(avgTilt) < 15)
    smileCount += 1 if get_expression(frame, landmarks) == "Smiling" else 0
    frameCount += 1
    putTextOnImage(
        data_img, f"Smiling : {round(smileCount / frameCount * 100)}%", 350, True
    )
    total_sec = int(time.time() - start_time)
    hour = total_sec // 3600
    min = (total_sec % 3600) // 60
    sec = total_sec % 60

    putTextOnImage(data_img, f"Session Time: {hour}h {min}m {sec}s", 400, True)


data_img = np.zeros((450, 600, 3), dtype=np.uint8)
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
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
