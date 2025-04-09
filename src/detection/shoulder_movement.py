import cv2
import numpy as np
from collections import deque
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

left_shoulder_history = deque(maxlen=20)
right_shoulder_history = deque(maxlen=20)
neck_angle_history = deque(maxlen=10)  # Lịch sử góc cổ
SMOOTHING_WINDOW = 5

def smooth_history(history):
    if len(history) < SMOOTHING_WINDOW:
        return list(history)
    smoothed = np.convolve(history, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW, mode='valid')
    return smoothed

def estimate_shoulder_positions(frame):
    h, w = frame.shape[:2]
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None, None, None
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    return (left_shoulder.x * w, left_shoulder.y * h), (right_shoulder.x * w, right_shoulder.y * h), (nose.x * w, nose.y * h)

def detect_shoulder_behaviors(frame):
    suspicious_score = 0
    reasons = []
    h, w = frame.shape[:2]

    left_shoulder, right_shoulder, nose_position = estimate_shoulder_positions(frame)
    if left_shoulder and right_shoulder and nose_position:
        left_x, left_y = left_shoulder
        right_x, right_y = right_shoulder
        nose_x, nose_y = nose_position

        left_shoulder_history.append((left_x, left_y))
        right_shoulder_history.append((right_x, right_y))

        # Tính góc cổ
        shoulder_mid_x = (left_x + right_x) / 2
        shoulder_mid_y = (left_y + right_y) / 2
        neck_angle = np.arctan2(shoulder_mid_y - nose_y, shoulder_mid_x - nose_x) * 180 / np.pi
        neck_angle_history.append(neck_angle)

        # ==== Vai di chuyển bất thường (qua lại nhanh) ====
        if len(left_shoulder_history) >= 10 and len(right_shoulder_history) >= 10:
            left_x_positions = [pos[0] for pos in left_shoulder_history]
            right_x_positions = [pos[0] for pos in right_shoulder_history]
            smoothed_left_x = smooth_history(left_x_positions)
            smoothed_right_x = smooth_history(right_x_positions)

            left_range = abs(max(smoothed_left_x) - min(smoothed_left_x))
            right_range = abs(max(smoothed_right_x) - min(smoothed_right_x))
            if left_range > 0.15 * w or right_range > 0.15 * w:
                suspicious_score += 2
                reasons.append("Shoulder moving suspiciously")

        # ==== Cổ xoay nhanh, nghiêng bất thường hoặc cúi đầu ====
        if len(neck_angle_history) >= 10:
            smoothed_angles = smooth_history(neck_angle_history)
            angle_velocity = abs(max(smoothed_angles) - min(smoothed_angles)) / (len(smoothed_angles) * (1/30))  # Giả định 30 FPS
            if angle_velocity > 50:  # Tốc độ thay đổi góc > 50 độ/giây
                suspicious_score += 2
                reasons.append(f"Neck rotating rapidly (vel: {angle_velocity:.1f} deg/s)")
            elif abs(neck_angle) > 100:  # Ngưỡng nghiêng tĩnh
                suspicious_score += 2
                reasons.append(f"Neck tilted abnormally (angle: {neck_angle:.1f} deg)")
            elif neck_angle < 80 or  130> neck_angle > 180 :  # Phát hiện cúi đầu khi góc cổ < 20 độ
                suspicious_score += 1
                reasons.append(f"Head pitch down (angle: {neck_angle:.1f} deg)")

    return suspicious_score, reasons