import cv2
import mediapipe as mp
import math
import time
import os
import json
import serial
from datetime import datetime, timedelta

# === CONFIGURATION ===
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 320
EAR_THRESHOLD = 0.17
SLEEP_TRIGGER_TIME = 1.25
MAX_RECORDING_DURATION = 300
SERIAL_PORT = 'COM3'  # Change this to your Arduino COM port
BAUD_RATE = 9600

# === FILE SETUP ===
SAVE_PATH = "saved"
VIDEO_FOLDER = os.path.join(SAVE_PATH, "video")
JSON_PATH = os.path.join(SAVE_PATH, "main.json")

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

# === ARDUINO SERIAL SETUP ===
def update_arduino_status():
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)  # Let Arduino reset
        arduino_connected = True
        print("[INFO] Arduino connected on", SERIAL_PORT)
    except:
        arduino = None
        arduino_connected = False
        print("[WARNING] No Arduino device detected on", SERIAL_PORT)

    if not os.path.exists(JSON_PATH) or os.stat(JSON_PATH).st_size == 0:
        data = {"accident": {}, "connected-ino": arduino_connected}
    else:
        with open(JSON_PATH, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"accident": {}}
        data["connected-ino"] = arduino_connected

    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    return arduino, arduino_connected

arduino, arduino_connected = update_arduino_status()

# === MEDIAPIPE SETUP ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices):
    A = math.dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = math.dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = math.dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (A + B) / (2.0 * C)

def create_accident_entry(counter, sleep_time, wake_time=None, video_path=""):
    duration = (
        str(timedelta(seconds=int(wake_time - sleep_time)))
        if wake_time else "00:00:00"
    )
    opened = bool(wake_time)
    dt_str = datetime.fromtimestamp(sleep_time).strftime("%d/%m/%Y %H:%M:%S")
    key = f"accident-{counter}"

    with open(JSON_PATH, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"accident": {}}

    data["accident"][key] = {
        "time": dt_str,
        "opened": opened,
        "how-long": duration,
        "video-path": video_path.replace("\\", "/")
    }

    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

def delete_old_videos(video_folder, days_old=30):
    now = time.time()
    for fname in os.listdir(video_folder):
        fpath = os.path.join(video_folder, fname)
        if os.path.isfile(fpath):
            ctime = os.path.getctime(fpath)
            if (now - ctime) > (days_old * 86400):
                os.remove(fpath)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

closed_start_time = None
recording = False
video_writer = None
video_start_time = None
accident_counter = 1
video_filename = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        both_closed = left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD

        for idx in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)

        if both_closed:
            if closed_start_time is None:
                closed_start_time = time.time()
            else:
                duration = time.time() - closed_start_time
                if duration >= SLEEP_TRIGGER_TIME:
                    cv2.putText(frame, "ALERT: Don't Close Your Eyes!", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    print("ALERT: Don't Close Your Eyes!", end="\r")

                    if not recording:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_filename = os.path.join(VIDEO_FOLDER, f"{timestamp}.avi")
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (w, h))
                        video_start_time = time.time()
                        accident_start_time = video_start_time
                        recording = True
                        if arduino:
                            arduino.write(b'1')
        else:
            if recording:
                wake_time = time.time()
                create_accident_entry(accident_counter, accident_start_time, wake_time, video_filename)
                accident_counter += 1
                video_writer.release()
                video_writer = None
                recording = False
                if arduino:
                    arduino.write(b'0')
            closed_start_time = None

        status_text = "Eyes Closed" if both_closed else "Eyes Open"
        color = (0, 0, 255) if both_closed else (0, 255, 0)
        cv2.putText(frame, status_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if recording and video_writer:
        timestamp_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        text_size, _ = cv2.getTextSize(timestamp_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = w - text_size[0] - 10
        text_y = 30
        cv2.putText(frame, timestamp_str, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        video_writer.write(frame)

        if time.time() - video_start_time >= MAX_RECORDING_DURATION:
            create_accident_entry(accident_counter, accident_start_time, None, video_filename)
            accident_counter += 1
            video_writer.release()
            video_writer = None
            recording = False

            if arduino:
                arduino.write(b'0')

    cv2.imshow("Eye Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
if video_writer:
    video_writer.release()

    
cv2.destroyAllWindows()
delete_old_videos(VIDEO_FOLDER, days_old=30)
