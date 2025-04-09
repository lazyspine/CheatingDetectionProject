import mediapipe as mp
import cv2
import numpy as np
from collections import deque

# Khởi tạo Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Các điểm landmark quan trọng
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133]      # Khóe mắt trái
RIGHT_EYE = [362, 263]    # Khóe mắt phải
LEFT_EYE_VERT = [159, 145]  # Điểm trên và dưới mắt trái để phát hiện nháy mắt
RIGHT_EYE_VERT = [386, 374]  # Điểm trên và dưới mắt phải

# Lịch sử hướng nhìn
gaze_history = deque(maxlen=10)  # Lưu 10 khung hình gần nhất

def preprocess_frame(frame):
    """Tiền xử lý ảnh để tăng độ chính xác."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return img_rgb

def get_eye_gaze_direction(frame):
    h, w = frame.shape[:2]
    img_rgb = preprocess_frame(frame)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "center", None, None  # Trả về thêm tọa độ để visualize

    face = results.multi_face_landmarks[0]

    # Trích xuất tọa độ và visibility
    def get_coords(indexes):
        coords = []
        visibilities = []
        for i in indexes:
            lm = face.landmark[i]
            coords.append([int(lm.x * w), int(lm.y * h)])
            visibilities.append(lm.visibility)
        return np.array(coords), np.mean(visibilities)  # Trung bình visibility

    left_eye, left_eye_vis = get_coords(LEFT_EYE)
    right_eye, right_eye_vis = get_coords(RIGHT_EYE)
    left_iris, left_iris_vis = get_coords(LEFT_IRIS)
    right_iris, right_iris_vis = get_coords(RIGHT_IRIS)
    left_eye_vert, _ = get_coords(LEFT_EYE_VERT)  # Để phát hiện nháy mắt
    right_eye_vert, _ = get_coords(RIGHT_EYE_VERT)

    # Làm mịn tọa độ
    def smooth_coords(coords):
        return np.mean(coords, axis=0)

    left_eye_center = smooth_coords(left_eye)
    right_eye_center = smooth_coords(right_eye)
    left_iris_center = smooth_coords(left_iris)
    right_iris_center = smooth_coords(right_iris)

    # Phát hiện nháy mắt
    def is_blinking(vert_coords):
        eye_height = vert_coords[1][1] - vert_coords[0][1]  # Khoảng cách dọc
        return eye_height < 10  # Ngưỡng nháy mắt (có thể điều chỉnh)

    left_blink = is_blinking(left_eye_vert)
    right_blink = is_blinking(right_eye_vert)
    # if left_blink or right_blink:
    #     return "blink", left_eye_center, right_eye_center  # Trả về nếu nháy mắt

    # Điều chỉnh ngưỡng động dựa trên visibility
    base_hor_threshold = 0.20
    base_vert_threshold = 0.15
    vis_threshold_factor = min(left_eye_vis, right_eye_vis, left_iris_vis, right_iris_vis)
    hor_threshold = base_hor_threshold * (1 + (1 - vis_threshold_factor))  # Giảm độ nhạy nếu visibility thấp
    vert_threshold = base_vert_threshold * (1 + (1 - vis_threshold_factor))

    # Phân tích hướng ngang
    def analyze_horizontal_gaze(eye_center, iris_center, eye_width):
        rel_x = (iris_center[0] - eye_center[0]) / (eye_width + 1e-6)
        if rel_x < -hor_threshold:
            return "left"
        elif rel_x > hor_threshold:
            return "right"
        return "center"

    # Phân tích hướng dọc
    def analyze_vertical_gaze(eye_center, iris_center, eye_height):
        rel_y = (iris_center[1] - eye_center[1]) / (eye_height + 1e-6)
        if rel_y < -vert_threshold:
            return "down"
        elif rel_y > vert_threshold:
            return "up"
        return "center"

    eye_width_left = np.linalg.norm(left_eye[1] - left_eye[0])
    eye_width_right = np.linalg.norm(right_eye[1] - right_eye[0])
    eye_height_left = eye_width_left * 0.5
    eye_height_right = eye_width_right * 0.5

    hor_left = analyze_horizontal_gaze(left_eye_center, left_iris_center, eye_width_left)
    hor_right = analyze_horizontal_gaze(right_eye_center, right_iris_center, eye_width_right)
    vert_left = analyze_vertical_gaze(left_eye_center, left_iris_center, eye_height_left)
    vert_right = analyze_vertical_gaze(right_eye_center, right_iris_center, eye_height_right)

    # Quyết định hướng
    if hor_left == hor_right and hor_left != "center":
        gaze = hor_left
    elif vert_left == vert_right and vert_left != "center":
        gaze = vert_left
    else:
        gaze = "center"

    # Bộ lọc thời gian
    gaze_history.append(gaze)
    if len(gaze_history) >= 10:
        gaze_counts = {g: gaze_history.count(g) for g in set(gaze_history)}
        gaze = max(gaze_counts, key=gaze_counts.get)  # Lấy hướng phổ biến nhất

    return gaze, left_eye_center, right_eye_center

# Ví dụ sử dụng với visualization
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gaze, left_center, right_center = get_eye_gaze_direction(frame)

        # Visualization
        if left_center is not None and right_center is not None:
            left_x, left_y = int(left_center[0]), int(left_center[1])
            right_x, right_y = int(right_center[0]), int(right_center[1])
            # Vẽ trung tâm mắt
            cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)
            # Vẽ đường nối hai mắt
            cv2.line(frame, (left_x, left_y), (right_x, right_y), (255, 0, 0), 2)

        cv2.putText(frame, f"Gaze: {gaze}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Eye Gaze Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()