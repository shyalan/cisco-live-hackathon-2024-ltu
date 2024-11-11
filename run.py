import cv2
import mediapipe as mp
import time
from ultralytics import YOLO

model = YOLO('yolov5s.pt')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(1) 

OBJECT_CLASSES = [67]

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = model(frame, conf=0.25)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb_frame)
    result_face = face_mesh.process(rgb_frame)
    eye_position = None
    if result_face.multi_face_landmarks:
        for face_landmarks in result_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            )
            h, w, _ = frame.shape
            left_eye_x = int(face_landmarks.landmark[133].x * w)
            left_eye_y = int(face_landmarks.landmark[133].y * h)
            right_eye_x = int(face_landmarks.landmark[362].x * w)
            right_eye_y = int(face_landmarks.landmark[362].y * h)
            eye_position = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
            cv2.circle(frame, (left_eye_x, left_eye_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 5, (255, 0, 0), -1)
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
            elif class_id in OBJECT_CLASSES:
                objects.append((x1, y1, x2, y2, class_id))
    if best_person:
        x1, y1, x2, y2 = best_person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person: {highest_confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
                    if eye_position and ox1 <= eye_position[0] <= ox2 and oy1 <= eye_position[1] <= oy2:
                        if not looking_at_object:
                            looking_at_object = True
                            gaze_start_time = time.time()
                        else:
                            gaze_timer = time.time() - gaze_start_time
                    else:
                        looking_at_object = False
                        gaze_timer = 0
                    if gaze_timer > 0:
                        cv2.putText(frame, f"Gaze Time: {gaze_timer:.2f}s", (ox1, oy1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow('Real-Time Person-Object and Eye Tracking with Timer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
