import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

KEY_POINTS = [1, 152, 33, 263, 61, 291]

def estimate_head_direction(frame, face_bbox=None):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return "unknown"

    face_landmarks = results.multi_face_landmarks[0]
    h, w = frame.shape[:2]

    image_points = np.array([
        [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in KEY_POINTS
    ], dtype="double")

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1]
    ])

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rvec_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = euler_angles.flatten()

    if yaw < -150:
        return "left"
    elif yaw > 150:
        return "right"
    elif pitch < -150:
        return "up"
    elif pitch > 150:
        return "down"
    else:
        return "center"
