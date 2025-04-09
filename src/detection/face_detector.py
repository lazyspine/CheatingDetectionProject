import mediapipe as mp
import cv2
import numpy as np
from collections import deque

# Khởi tạo Mediapipe Face Detection
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(
    model_selection=0,  # 0: short-range (gần), 1: full-range (xa)
    min_detection_confidence=0.5  # Ngưỡng tin cậy tối thiểu
)

# Lịch sử phát hiện khuôn mặt để làm mịn
face_history = deque(maxlen=5)

def preprocess_frame(frame):
    """Tiền xử lý ảnh để tăng độ chính xác."""
    # Chuyển sang grayscale để giảm nhiễu màu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Tăng độ tương phản
    gray = cv2.equalizeHist(gray)
    # Làm mờ nhẹ để giảm nhiễu
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Chuyển lại RGB cho Mediapipe
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return img_rgb

def detect_faces(frame):
    """
    Phát hiện khuôn mặt trong khung hình.
    Trả về danh sách các bounding box (x, y, w, h) đã được làm mịn.
    """
    h, w = frame.shape[:2]
    # Tiền xử lý khung hình
    img_rgb = preprocess_frame(frame)
    results = detector.process(img_rgb)

    faces = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            # Chuyển tọa độ tương đối sang tuyệt đối
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            # Đảm bảo tọa độ không vượt khung hình
            x = max(0, x)
            y = max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)
            faces.append((x, y, width, height))  # Trả về tuple thay vì đối tượng bbox

    # Làm mịn kết quả bằng lịch sử
    if faces:
        face_history.append(faces[0])  # Chỉ theo dõi khuôn mặt đầu tiên (giả định 1 người)
    if not face_history:  # Nếu không phát hiện khuôn mặt
        return []

    # Tính trung bình tọa độ từ lịch sử
    smoothed_faces = []
    if len(face_history) >= 3:  # Cần ít nhất 3 khung để làm mịn
        avg_x = int(np.mean([f[0] for f in face_history]))
        avg_y = int(np.mean([f[1] for f in face_history]))
        avg_w = int(np.mean([f[2] for f in face_history]))
        avg_h = int(np.mean([f[3] for f in face_history]))
        # Đảm bảo tọa độ hợp lệ
        avg_x = max(0, avg_x)
        avg_y = max(0, avg_y)
        avg_w = min(w - avg_x, avg_w)
        avg_h = min(h - avg_y, avg_h)
        smoothed_faces.append((avg_x, avg_y, avg_w, avg_h))
    else:
        smoothed_faces = [face_history[-1]]  # Dùng khung gần nhất nếu chưa đủ lịch sử

    return smoothed_faces

# Ví dụ sử dụng
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()