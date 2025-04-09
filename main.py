import cv2
import time
from detection.face_detector import detect_faces
from detection.eye_gaze import get_eye_gaze_direction
from detection.face_orientation import estimate_face_orientation, estimate_head_pitch
from utils.logger import log_event, log_status_realtime
from utils.alert import alert

cap = cv2.VideoCapture(0)
current_status = "Normal"
suspicion_accum = 0
over_threshold_start = None
THRESHOLD = 12
DURATION = 6

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

    # ==== 2. Eye Gaze Detection ====
    eye_dir = get_eye_gaze_direction(frame)
    if eye_dir in ["left", "right"]:
        suspicious_score += 1
        reasons.append(f"Eye gaze {eye_dir}")
    elif eye_dir in ["up", "down"]:
        suspicious_score += 1
        reasons.append(f"Eye gaze {eye_dir}")

    # ==== 3. Face Orientation (quay trái/phải) ====
    face_dir = estimate_face_orientation(frame)
    if face_dir in ["left", "right"]:
        suspicious_score += 1
        reasons.append(f"Face turned {face_dir}")

    # ==== 4. Head Pitch (cúi/ngẩng mặt) ====
    pitch_dir = estimate_head_pitch(frame)
    if pitch_dir in ["up", "down"]:
        suspicious_score += 1
        reasons.append(f"Head pitch {pitch_dir}")

    # ==== 5. Tích điểm nghi ngờ ====
    if suspicious_score > 0:
        suspicion_accum += suspicious_score
    else:
        suspicion_accum = max(0, suspicion_accum - 1)

    # ==== 6. Phân mức cảnh báo ====
    if suspicion_accum == 0:
        level = "✅ Normal"
    elif suspicion_accum < 40:
        level = "⚠️ Low Suspicion"
    elif suspicion_accum < 80:
        level = "❗ Medium Suspicion"
    else:
        level = "🚨 High Suspicion"

    status = f"{level}: {', '.join(reasons) if reasons else 'No issues'}"

    # ==== 7. Hiển thị lên webcam ====
    cv2.putText(frame, f"Suspicion Score: {suspicion_accum}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255) if suspicion_accum >= 10 else
                     (0, 165, 255) if suspicion_accum >= 5 else
                     (0, 255, 255) if suspicion_accum > 0 else
                     (0, 255, 0), 2)

    # ==== 8. Cảnh báo nếu duy trì nghi ngờ ====
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

    # ==== 9. Log nếu trạng thái thay đổi ====
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
