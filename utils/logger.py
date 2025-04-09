import csv
import os
from datetime import datetime
import cv2

LOG_FILE = "data/logs.csv"
EVIDENCE_DIR = "data/evidence"

def log_event(message, frame=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, message])
    if frame is not None:
        if not os.path.exists(EVIDENCE_DIR):
            os.makedirs(EVIDENCE_DIR)
        filename = os.path.join(EVIDENCE_DIR, f"{timestamp}.jpg")
        cv2.imwrite(filename, frame)

def log_status_realtime(status, score=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, status, score])
    print(f"{timestamp} | {status} | Score: {score}")
