import cv2
import dlib
import numpy as np
import winsound
import threading
from collections import deque
import os

def alert_driver():
    """Plays a loud wav file. Ensure 'alarm.wav' is in the same folder."""
    if os.path.exists("alarm.wav"):
        # Play asynchronously to avoid freezing video
        winsound.PlaySound("alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        # Fallback to very loud high-pitched beep
        for _ in range(5):
            winsound.Beep(3500, 300)

# --- CONFIGURATION ---
EAR_THRESHOLD = 0.18          # Lowered: Only triggers on full closure
CLOSED_FRAMES_LIMIT = 40        # Adjusted for higher precision
MOUTH_YAWN_THRESHOLD = 0.60  # Mouth Aspect Ratio threshold
CONSECUTIVE_FRAMES = 40      # Frames eyes must stay closed
EYE_HISTORY = deque(maxlen=300)



def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_mouth_aspect_ratio(landmarks):
    # Using specific points for inner lips
    top_lip = (landmarks[50] + landmarks[51] + landmarks[52]) / 3
    bottom_lip = (landmarks[58] + landmarks[57] + landmarks[56]) / 3
    left_corner = landmarks[48]
    right_corner = landmarks[54]
    
    vertical = euclidean_dist(top_lip, bottom_lip)
    horizontal = euclidean_dist(left_corner, right_corner)
    return vertical / horizontal

def get_head_pose(shape, frame_shape):
    # 2D image points from dlib landmarks
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye left corner
        (shape.part(45).x, shape.part(45).y),     # Right eye right corner
        (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype="double")

    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    size = frame_shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    return rotation_vector, translation_vector

# --- INITIALIZATION ---
predictor_path = r"C:\Users\Aryan\Desktop\Blind Drowness\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)
COUNTER = 0
ALARM_ON = False
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    faces = detector(gray, 0)
    
    
    
            
    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in shape.parts()])
        
        rot_vec, _ = get_head_pose(shape, frame.shape)
        pitch = rot_vec[0][0] * (180/np.pi)
        yaw = rot_vec[1][0] * (180/np.pi)

# IMPROVED EAR: Use the max of both eyes. 
# If you turn your head, one eye usually stays 'more open' to the camera.
        left_ear = eye_aspect_ratio(landmarks[36:42])
        right_ear = eye_aspect_ratio(landmarks[42:48])
        ear = max(left_ear, right_ear)        
        mouth = landmarks[48:68]
        
        
        
        mar = get_mouth_aspect_ratio(landmarks)

        # Logic 1: Drowsiness Detection
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (100, 200),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)
                if not ALARM_ON:
                    ALARM_ON = True
                    # Use a daemon thread so it dies when the main program closes
                    t = threading.Thread(target=alert_driver)
                    t.daemon = True
                    t.start()
        else:
            COUNTER = 0
            ALARM_ON = False
        
        # Check for looking Left/Right (Yaw) OR Up/Down (Pitch)
        # yaw > 20: looking at mirrors | pitch > 15: looking at phone
        if abs(yaw) > 20 or abs(pitch) > 15: 
            cv2.putText(frame, "DISTRACTION ALERT!", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
            
        # Logic 2: Yawn Detection (Visual Alert)
        if mar > MOUTH_YAWN_THRESHOLD:
            cv2.putText(frame, "YAWNING DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # Logic 3: PERCLOS Calculation
        EYE_HISTORY.append(1 if ear < EAR_THRESHOLD else 0)
        perclos = (sum(EYE_HISTORY) / len(EYE_HISTORY)) * 100

        # Draw visual markers
        for (x, y) in landmarks[36:68]:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # HUD (Heads Up Display)
        # HUD (Heads Up Display)
        cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f} | Pitch: {pitch:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "NO FACE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Driver Safety System", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()