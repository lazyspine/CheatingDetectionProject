import mediapipe as mp
import cv2
import numpy as np

# Khởi tạo Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Các điểm landmark quan trọng
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133]      # khóe mắt trái
RIGHT_EYE = [362, 263]    # khóe mắt phải

def get_eye_gaze_direction(frame):
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "center"

    face = results.multi_face_landmarks[0]

    # Trích xuất tọa độ điểm landmark
    def get_coords(indexes):
        return np.array([
            [int(face.landmark[i].x * w), int(face.landmark[i].y * h)]
            for i in indexes
        ])

    left_eye = get_coords(LEFT_EYE)
    right_eye = get_coords(RIGHT_EYE)
    left_iris = get_coords(LEFT_IRIS)
    right_iris = get_coords(RIGHT_IRIS)

    # Phân tích tròng mắt nằm lệch trái/phải
    def analyze_gaze(eye_pts, iris_pts):
        eye_left, eye_right = eye_pts
        iris_center = np.mean(iris_pts, axis=0)
        eye_width = eye_right[0] - eye_left[0]
        rel_x = (iris_center[0] - eye_left[0]) / (eye_width + 1e-6)

        if rel_x < 0.30:       # tròng mắt lệch trái
            return "left"
        elif rel_x > 0.70:     # lệch phải
            return "right"
        else:
            return "center"

    # Phân tích tròng mắt lệch trên/dưới (cúi đầu/ngẩng mặt)
    def vertical_estimate(eye_pts, iris_pts):
        eye_top = eye_pts[0][1]
        eye_bottom = eye_pts[1][1]
    
        if eye_bottom - eye_top < 5:  # kiểm tra tránh chia gần 0
            return "center"
    
        iris_center = np.mean(iris_pts, axis=0)
        rel_y = (iris_center[1] - eye_top) / (eye_bottom - eye_top)
    
        # DEBUG
        print(f"eye_top: {eye_top}, eye_bottom: {eye_bottom}, rel_y: {rel_y:.3f}")
    
        if rel_y < 0.15:
            return "down"
        elif rel_y > 0.55:
            return "up"
        else:
            return "center"


    hor_left = analyze_gaze(left_eye, left_iris)
    hor_right = analyze_gaze(right_eye, right_iris)
    vert = vertical_estimate(left_eye, left_iris)

    # Nếu hai mắt cùng hướng ngang → ưu tiên dùng
    if hor_left == hor_right:
        if vert != "center":
            return vert
        return hor_left

    return vert if vert != "center" else "center"
