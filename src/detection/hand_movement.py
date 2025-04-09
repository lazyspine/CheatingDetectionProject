import cv2
import numpy as np
from collections import deque
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

hand_history = deque(maxlen=20)
SMOOTHING_WINDOW = 5

def smooth_history(history):
    if len(history) < SMOOTHING_WINDOW:
        return list(history)
    smoothed = np.convolve(history, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode='valid')
    return smoothed

def estimate_hand_position(frame):
    """Ước lượng vị trí tay và mũi."""
    h, w = frame.shape[:2]
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None, None, None  # Trả về None nếu không phát hiện được
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    return (left_wrist.x * w, left_wrist.y * h), (right_wrist.x * w, right_wrist.y * h), (nose.x * w, nose.y * h)

def detect_hand_behaviors(frame):
    """Phát hiện các hành vi đáng ngờ liên quan đến tay."""
    suspicious_score = 0
    reasons = []
    h, w = frame.shape[:2]

    left_wrist, right_wrist, nose_position = estimate_hand_position(frame)
    if left_wrist and right_wrist and nose_position:  # Kiểm tra nếu tất cả đều có giá trị
        hand_x = min(left_wrist[0], right_wrist[0])
        hand_y = min(left_wrist[1], right_wrist[1])
        nose_x, nose_y = nose_position
        hand_history.append((hand_x, hand_y))

        # ==== Tay ra ngoài khung hình (có thể lấy tài liệu bên ngoài) ====
        if hand_x < 0.1 * w or hand_x > 0.9 * w or hand_y < 0.1 * h:
            suspicious_score += 2
            reasons.append("Hand out of frame")

        # ==== Tay gần tai/miệng (có thể đang nghe hoặc nói gì đó) ====
        face_region_y = nose_y - 0.1 * h  # Vùng gần mặt dựa trên mũi
        face_region_x = nose_x
        hand_near_face = abs(hand_x - face_region_x) < 0.1 * w and hand_y < face_region_y
        if hand_near_face and len(hand_history) >= 10:
            recent_positions = list(hand_history)[-10:]
            if all(abs(pos[0] - face_region_x) < 0.1 * w and pos[1] < face_region_y for pos in recent_positions):
                suspicious_score += 3
                reasons.append("Hand near ear/mouth")

        # ==== Tay di chuyển bất thường (qua lại nhanh gần vùng ngoài) ====
        if len(hand_history) >= 10:
            x_positions = [pos[0] for pos in hand_history]
            smoothed_x = smooth_history(x_positions)
            x_range = abs(max(smoothed_x) - min(smoothed_x))
            if x_range > 0.3 * w and (min(smoothed_x) < 0.2 * w or max(smoothed_x) > 0.8 * w):
                suspicious_score += 2
                reasons.append("Hand moving suspiciously near edges")

    return suspicious_score, reasons