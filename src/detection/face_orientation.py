import cv2
import numpy as np
from collections import deque
import mediapipe as mp

# Khởi tạo module Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Bộ đệm lịch sử
orientation_history = deque(maxlen=30)  # Lịch sử hướng khuôn mặt
pitch_history = deque(maxlen=30)        # Lịch sử pitch
SMOOTHING_WINDOW = 5                    # Cửa sổ làm mịn

def smooth_history(history):
    """Làm mịn dữ liệu lịch sử để giảm nhiễu."""
    if len(history) < SMOOTHING_WINDOW:
        return list(history)
    smoothed = np.convolve(history, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode='valid')
    return smoothed

def estimate_face_orientation(frame):
    """Ước lượng hướng khuôn mặt (trái, phải, giữa)."""
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "unknown"

    face = results.multi_face_landmarks[0]
    nose = face.landmark[1]
    left_eye = face.landmark[33]
    right_eye = face.landmark[263]

    nose_x = nose.x * w
    left_dist = nose_x - (left_eye.x * w)
    right_dist = (right_eye.x * w) - nose_x

    threshold = 0.08 * w
    if left_dist - right_dist > threshold:
        return "left"
    elif right_dist - left_dist > threshold:
        return "right"
    else:
        return "center"

def estimate_head_pitch(frame):
    """Ước lượng độ nghiêng đầu (ngẩng, cúi, giữa)."""
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "center"

    face = results.multi_face_landmarks[0]
    nose_tip = face.landmark[1]
    left_eye_top = face.landmark[159]
    right_eye_top = face.landmark[386]

    eye_avg_y = (left_eye_top.y + right_eye_top.y) / 2 * h
    nose_y = nose_tip.y * h
    delta = nose_y - eye_avg_y

    if delta < 0.05 * h:
        return "up"
    elif delta > 0.2 * h:
        return "down"
    else:
        return "center"

def detect_head_behaviors(frame):
    """Phát hiện các hành vi đáng ngờ liên quan đến đầu."""
    suspicious_score = 0
    reasons = []

    # ==== Hướng khuôn mặt (trái/phải) + Lắc đầu ====
    face_dir = estimate_face_orientation(frame)
    orientation_history.append(face_dir)
    if face_dir in ["left", "right"]:
        suspicious_score += 1
        reasons.append(f"Face turned {face_dir}")
    
    smoothed_orient = smooth_history([1 if o == "left" else -1 if o == "right" else 0 for o in orientation_history])
    if len(smoothed_orient) >= 10 and np.max(smoothed_orient) > 0.5 and np.min(smoothed_orient) < -0.5:
        suspicious_score += 2
        reasons.append("Shaking head")

    # ==== Độ nghiêng đầu (ngẩng/cúi) + Gật đầu ====
    pitch_dir = estimate_head_pitch(frame)
    pitch_history.append(pitch_dir)
    if pitch_dir in ["up", "down"]:
        suspicious_score += 1
        reasons.append(f"Head pitch {pitch_dir}")
    
    smoothed_pitch = smooth_history([1 if p == "up" else -1 if p == "down" else 0 for p in pitch_history])
    if len(smoothed_pitch) >= 10 and np.max(smoothed_pitch) > 0.5 and np.min(smoothed_pitch) < -0.5:
        suspicious_score += 2
        reasons.append("Nodding")

    return suspicious_score, reasons