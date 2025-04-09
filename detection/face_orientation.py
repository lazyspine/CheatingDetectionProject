import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark quan trọng:
# Nose: 1, Left eye: 33, Right eye: 263

def estimate_face_orientation(frame):
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "unknown"

    face = results.multi_face_landmarks[0]
    nose = face.landmark[1]
    left_eye = face.landmark[33]
    right_eye = face.landmark[263]

    nose_x = nose.x
    left_dist = nose_x - left_eye.x
    right_dist = right_eye.x - nose_x

    if left_dist - right_dist > 0.08:
        return "left"
    elif right_dist - left_dist > 0.08:
        return "right"
    else:
        return "center"
def estimate_head_pitch(frame):
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return "center"

    face = results.multi_face_landmarks[0]
    nose_tip = face.landmark[1]
    left_eye_top = face.landmark[159]   # mắt trái trên
    right_eye_top = face.landmark[386]  # mắt phải trên

    eye_avg_y = (left_eye_top.y + right_eye_top.y) / 2
    delta = nose_tip.y - eye_avg_y  # nếu delta âm => ngẩng mặt

    # DEBUG
    # print(f"delta_pitch: {delta:.3f}")

    if delta < 0.05:   # ngẩng mặt cao
        return "up"
    elif delta > 0.2:  # cúi mặt sâu
        return "down"
    else:
        return "center"