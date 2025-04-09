import mediapipe as mp
import cv2

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_faces(frame):
    results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            faces.append(bbox)
    return faces
