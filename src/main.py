import cv2
import time
import numpy as np
from detection.face_detector import detect_faces
from detection.eye_gaze import get_eye_gaze_direction  # Đảm bảo dùng phiên bản trả về tọa độ
from detection.face_orientation import estimate_face_orientation, estimate_head_pitch, detect_head_behaviors
from detection.shoulder_movement import detect_shoulder_behaviors, estimate_shoulder_positions
from utils.logger import log_event, log_status_realtime
from utils.alert import alert

cap = cv2.VideoCapture(0)
current_status = "Normal"
suspicion_accum = 0
over_threshold_start = None
THRESHOLD = 12
DURATION = 6

neck_angle_history = []  # Để tính tốc độ trong main

while True:
    ret, frame = cap.read()
    if not ret:
        break
    og_frame = frame
    faces = detect_faces(frame)
    suspicious_score = 0
    reasons = []

    # ==== 1. Check số khuôn mặt ====
    if len(faces) == 0:
        suspicious_score += 2
        reasons.append("No face detected")
    elif len(faces) > 1:
        suspicious_score += 3
        reasons.append("Multiple faces")

    # ==== 2. Eye Gaze Detection với visualization ====
    eye_dir, left_eye_center, right_eye_center = get_eye_gaze_direction(frame)
    if eye_dir in ["left", "right"]:
        suspicious_score += 1
        reasons.append(f"Eye gaze {eye_dir}")
    elif eye_dir in ["up", "down"]:
        suspicious_score += 1
        reasons.append(f"Eye gaze {eye_dir}")
    elif eye_dir == "blink":
        reasons.append("Blinking detected")

    # Visualization cho mắt (màu vàng, không nối đường)
    if left_eye_center is not None and right_eye_center is not None:
        left_x, left_y = int(left_eye_center[0]), int(left_eye_center[1])
        right_x, right_y = int(right_eye_center[0]), int(right_eye_center[1])
        # Vẽ trung tâm mắt bằng màu vàng
        cv2.circle(frame, (left_x, left_y), 5, (255, 255, 0), -1)  # Mắt trái (vàng)
        cv2.circle(frame, (right_x, right_y), 5, (255, 255, 0), -1)  # Mắt phải (vàng)

    # ==== 3 & 4. Phát hiện hành vi đầu ====
    head_score, head_reasons = detect_head_behaviors(frame)
    suspicious_score += head_score
    reasons.extend(head_reasons)

    # ==== 5. Phát hiện hành vi vai ====
    shoulder_score, shoulder_reasons = detect_shoulder_behaviors(frame)
    suspicious_score += shoulder_score
    reasons.extend(shoulder_reasons)

    # Visualization cho vai và cổ
    left_shoulder, right_shoulder, nose_position = estimate_shoulder_positions(frame)
    if left_shoulder and right_shoulder and nose_position:
        left_x, left_y = int(left_shoulder[0]), int(left_shoulder[1])
        right_x, right_y = int(right_shoulder[0]), int(right_shoulder[1])
        nose_x, nose_y = int(nose_position[0]), int(nose_position[1])

        cv2.line(frame, (nose_x, nose_y), (left_x, left_y), (0, 255, 0), 2)  # Đường xanh lá
        cv2.line(frame, (nose_x, nose_y), (right_x, right_y), (255, 0, 0), 2)  # Đường xanh dương
        cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)  # Vai trái (xanh lá)
        cv2.circle(frame, (right_x, right_y), 5, (255, 0, 0), -1)  # Vai phải (xanh dương)
        cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)  # Mũi (đỏ)

        shoulder_mid_x = (left_x + right_x) / 2
        shoulder_mid_y = (left_y + right_y) / 2
        neck_angle = np.arctan2(shoulder_mid_y - nose_y, shoulder_mid_x - nose_x) * 180 / np.pi
        neck_angle_history.append(neck_angle)
        if len(neck_angle_history) > 10:
            neck_angle_history.pop(0)
            smoothed_angles = np.convolve(neck_angle_history, np.ones(5)/5, mode='valid')
            angle_velocity = abs(max(smoothed_angles) - min(smoothed_angles)) / (len(smoothed_angles) * (1/30))

        cv2.putText(frame, f"Neck Angle: {neck_angle:.1f} deg", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(neck_angle_history) > 10:
            cv2.putText(frame, f"Angle Velocity: {angle_velocity:.1f} deg/s", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ==== 6. Tích điểm nghi ngờ ====
    if suspicious_score > 0:
        suspicion_accum += suspicious_score
    else:
        suspicion_accum = max(0, suspicion_accum - 1)

    # ==== 7. Phân mức cảnh báo ====
    if suspicion_accum == 0:
        level = "✅ Normal"
    elif suspicion_accum < 50:
        level = "⚠️ Low Suspicion"
    elif suspicion_accum < 100:
        level = "❗ Medium Suspicion"
    else:
        level = "🚨 High Suspicion"

    status = f"{level}: {', '.join(reasons) if reasons else 'No issues'}"

    # ==== 8. Hiển thị lên webcam ====
    cv2.putText(frame, f"Suspicion Score: {suspicion_accum}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255) if suspicion_accum >= 10 else
                     (0, 165, 255) if suspicion_accum >= 5 else
                     (0, 255, 255) if suspicion_accum > 0 else
                     (0, 255, 0), 2)

    # ==== 9. Cảnh báo nếu duy trì nghi ngờ ====
    if suspicion_accum >= THRESHOLD:
        if over_threshold_start is None:
            over_threshold_start = time.time()
        elif time.time() - over_threshold_start >= DURATION:
            alert_msg = f"⚠️ Sustained High Suspicion ({suspicion_accum}) > {THRESHOLD} for {DURATION}s"
            log_event(alert_msg, frame)
            alert(alert_msg)
            over_threshold_start = None
    else:
        over_threshold_start = None

    # ==== 10. Log nếu trạng thái thay đổi ====
    if status != current_status:
        log_status_realtime(status, suspicion_accum)
        if suspicious_score > 0:
            log_event(status, frame)
        alert(status)
        current_status = status

    cv2.imshow("Anti-Cheat Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()