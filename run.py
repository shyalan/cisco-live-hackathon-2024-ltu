import cv2
import mediapipe as mp
import time
import uuid
from ultralytics import YOLO
import subprocess
import pyrebase
import os

# Load YOLO model
model = YOLO('yolov5s.pt')

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

def is_close(bbox1, bbox2, margin=30):
    x1_a, y1_a, x2_a, y2_a = bbox1
    x1_b, y1_b, x2_b, y2_b = bbox2
    return (
        abs(x1_a - x2_b) <= margin or abs(x2_a - x1_b) <= margin or
        abs(y1_a - y2_b) <= margin or abs(y2_a - y1_b) <= margin
    )

gaze_timer = 0
gaze_start_time = 0
looking_at_object = False

detection_counter = 1

def log_detection(start_time, end_time, duration):
    try:
        with open("raw_held_log.txt", "a") as file:
            formatted_entry = (f"{start_time}, {end_time}, {duration:.2f} seconds\n")
            file.write(formatted_entry)
            print(f"Logged: {formatted_entry}")  # Debugging output
    except Exception as e:
        print(f"Error writing to file: {e}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = model(frame, conf=0.25)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = hands.process(rgb_frame)

        hand_positions = []
        if result_hands.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result_hands.multi_hand_landmarks):
                handedness = result_hands.multi_handedness[idx].classification[0].label
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                wrist_x, wrist_y = int(w * wrist.x), int(h * wrist.y)
                index_x, index_y = int(w * index_finger_tip.x), int(h * index_finger_tip.y)
                hand_positions.append((wrist_x, wrist_y, index_x, index_y, handedness))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        persons = []
        objects = []
        highest_confidence = -1
        best_person = None
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                if class_id == 0:
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_person = (x1, y1, x2, y2)
                else:
                    objects.append((x1, y1, x2, y2, class_id))

        if best_person:
            x1, y1, x2, y2 = best_person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            persons.append((x1, y1, x2, y2))

        for (px1, py1, px2, py2) in persons:
            for (ox1, oy1, ox2, oy2, object_class_id) in objects:
                if is_close((px1, py1, px2, py2), (ox1, oy1, ox2, oy2)) or (
                    px1 <= ox1 <= px2 and px1 <= ox2 <= px2 and py1 <= oy1 <= py2 and py1 <= oy2 <= py2
                ):
                    held_by_hand = False
                    for (wrist_x, wrist_y, index_x, index_y, handedness) in hand_positions:
                        if is_close((wrist_x, wrist_y, index_x, index_y), (ox1, oy1, ox2, oy2)):
                            held_by_hand = True
                            break
                    if held_by_hand:
                        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 3)
                        cv2.putText(frame, "Object Held", (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        if not looking_at_object:
                            looking_at_object = True
                            gaze_start_time = time.time()
                            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(gaze_start_time))
                            print(f"Started holding")
                        else:
                            gaze_timer = time.time() - gaze_start_time
                            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            log_detection(start_time, end_time, gaze_timer)
                    else:
                        if looking_at_object and gaze_timer > 0:
                            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            log_detection(start_time, end_time, gaze_timer)
                        looking_at_object = False
                        gaze_timer = 0

        cv2.imshow('Real-Time Object and Hand Tracking', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            subprocess.run(["python", "sort.py"])
        elif key == ord('d'):
            subprocess.run(["python", "databaseUpdate.py"])

finally:
    if looking_at_object and gaze_timer > 0:
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_detection(start_time, end_time, gaze_timer)
    cap.release()
    cv2.destroyAllWindows()
