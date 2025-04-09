import cv2
import numpy as np
import time
from datetime import datetime

class ExamCheatingDetector:
    def __init__(self, video_source=0, sensitivity=500):
        # Khởi tạo camera hoặc video file
        self.cap = cv2.VideoCapture(video_source)
        self.sensitivity = sensitivity  # Ngưỡng phát hiện chuyển động
        self.frame_count = 0
        self.alert_cooldown = 10  # Thời gian chờ giữa các cảnh báo (giây)
        self.last_alert_time = 0
        
        # Khởi tạo frame nền ban đầu
        ret, self.background = self.cap.read()
        if not ret:
            raise ValueError("Không thể đọc nguồn video.")
        self.background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        self.background = cv2.GaussianBlur(self.background, (21, 21), 0)

    def detect_motion(self, frame):
        # Chuyển frame sang grayscale và làm mờ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Tính toán sự khác biệt với background
        frame_diff = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Tìm contours của các vùng chuyển động
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.sensitivity:
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Suspicious Motion", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return motion_detected, frame

    def detect_head_pose(self, frame):
        # Placeholder cho nhận diện tư thế đầu
        # Trong thực tế, cần mô hình như MediaPipe hoặc Dlib
        h, w = frame.shape[:2]
        cv2.putText(frame, "Head Pose Monitoring", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def evaluate_performance(self, true_positives, false_positives, total_frames):
        accuracy = true_positives / total_frames if total_frames > 0 else 0
        fpr = false_positives / total_frames if total_frames > 0 else 0
        return accuracy, fpr

    def run(self):
        true_positives = 0
        false_positives = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Hết video hoặc lỗi nguồn.")
                break
                
            self.frame_count += 1
            
            # Phát hiện chuyển động
            motion_detected, processed_frame = self.detect_motion(frame.copy())
            
            # Phát hiện tư thế đầu (placeholder)
            processed_frame = self.detect_head_pose(processed_frame)
            
            # Hiển thị cảnh báo trên màn hình
            current_time = time.time()
            if motion_detected and (current_time - self.last_alert_time) > self.alert_cooldown:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(processed_frame, f"ALERT: Suspicious Activity {timestamp}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.last_alert_time = current_time
                true_positives += 1
            elif motion_detected:
                false_positives += 1

            # Hiển thị frame
            cv2.imshow("Exam Monitoring", processed_frame)
            
            # Tính toán hiệu suất
            accuracy, fpr = self.evaluate_performance(true_positives, false_positives, self.frame_count)
            print(f"Frame: {self.frame_count}, Accuracy: {accuracy:.2f}, FPR: {fpr:.2f}", end='\r')
            
            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Giải phóng tài nguyên
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nFinal Accuracy: {accuracy:.2f}, False Positive Rate: {fpr:.2f}")
