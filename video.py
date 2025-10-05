import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True
)  # refine gives iris and better eye points
mp_draw = mp.solutions.drawing_utils
eye_closed = False
blink_count = 0
eye_closed_prev = False


def detectEyes(frame):
    global blink_count
    global eye_closed
    global eye_closed_prev
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_indices = [33, 133, 159, 145, 153, 133]  # example
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

            # top_right = face_landmarks.landmark[386]
            # bottom_right = face_landmarks.landmark[374]
            # left_corner_r = face_landmarks.landmark[362]
            # right_corner_r = face_landmarks.landmark[263]
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
            if not eye_closed_prev and eye_closed:
                blink_count += 1
                print(blink_count)
    eye_closed_prev = eye_closed


faceArea = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        faceArea = w * h
        cv2.putText(
            frame,
            "Width: " + str(w) + " Height:" + str(h) + " Size: " + str(w * h),
            (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Blinks: " + str(blink_count),
            (50, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if w * h > 250000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    detectEyes(frame)
    cv2.imshow("Faces Detected", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
