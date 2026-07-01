"""
detector_opencv.py - Fixed Drowsiness Detection with Correct EAR
Uses improved eye detection and proper aspect ratio calculation
"""

import cv2
import numpy as np
from collections import deque
import time
import math

class DrowsinessDetector:
    def __init__(self):
        print("🚗 Initializing Drowsiness Detector (Improved)...")
        
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Thresholds - Adjusted for better accuracy
        self.EAR_THRESHOLD = 0.25      # Lower = eyes closed
        self.CONSEC_FRAMES = 12        # Frames before alert
        self.MIN_EYE_AREA = 50         # Minimum eye area to consider
        
        # History
        self.ear_history = deque(maxlen=30)
        self.alert_frames = 0
        self.is_alert = False
        
        # Tracking variables
        self.prev_ear = 0.3
        self.smoothing = 0.7  # Smoothing factor
        
        print("✅ Detector initialized!")
        print(f"   EAR Threshold: {self.EAR_THRESHOLD}")
        print(f"   Alert after: {self.CONSEC_FRAMES} frames")
        print("   Press 'q' to quit\n")
    
    def calculate_ear_from_rect(self, eye_rect, face_h):
        """
        Calculate EAR from a single eye rectangle
        EAR = height / width, normalized by face size
        """
        (ex, ey, ew, eh) = eye_rect
        
        # Basic EAR: height / width
        if ew == 0:
            return 0.3
        
        raw_ear = eh / ew
        
        # Normalize by face height to make it scale-invariant
        normalized_ear = raw_ear * (face_h / 200)
        
        # Clamp to reasonable range
        ear = min(max(normalized_ear * 1.2, 0.05), 0.5)
        
        return ear
    
    def detect_and_classify_eyes(self, roi_gray, roi_color, face_h):
        """
        Detect eyes and classify them as open/closed
        Returns: ear_value, eye_status, eye_positions
        """
        # Try different detection parameters for better results
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=1.05, 
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(80, 80)
        )
        
        # If no eyes found, try with different parameters
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(15, 15)
            )
        
        if len(eyes) < 2:
            # Not enough eyes detected
            return 0.25, "UNKNOWN", []
        
        # Sort eyes by x position
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        
        # Filter out very small detections (likely false positives)
        valid_eyes = []
        for (ex, ey, ew, eh) in eyes_sorted:
            area = ew * eh
            if area > self.MIN_EYE_AREA:
                valid_eyes.append((ex, ey, ew, eh))
        
        if len(valid_eyes) < 2:
            return 0.25, "UNKNOWN", valid_eyes
        
        # Take the two largest eyes (should be left and right)
        eye_areas = [(ex, ey, ew, eh, ew*eh) for (ex, ey, ew, eh) in valid_eyes]
        eye_areas_sorted = sorted(eye_areas, key=lambda e: e[4], reverse=True)
        
        left_eye = eye_areas_sorted[0][:4]
        right_eye = eye_areas_sorted[1][:4]
        
        # Calculate EAR for each eye
        left_ear = self.calculate_ear_from_rect(left_eye, face_h)
        right_ear = self.calculate_ear_from_rect(right_eye, face_h)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2
        
        # Apply smoothing to reduce jitter
        ear = self.smoothing * self.prev_ear + (1 - self.smoothing) * ear
        self.prev_ear = ear
        
        # Determine eye state
        is_closed = ear < self.EAR_THRESHOLD
        status = "CLOSED" if is_closed else "OPEN"
        
        # Draw eyes with color
        for (ex, ey, ew, eh) in [left_eye, right_eye]:
            color = (0, 0, 255) if is_closed else (0, 255, 0)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            
            # Add EAR value on eye
            cv2.putText(roi_color, f"{ear:.2f}", (ex, ey-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return ear, status, [left_eye, right_eye]
    
    def process_frame(self, frame):
        """Process a single frame"""
        if frame is None or frame.size == 0:
            return frame, False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame, self.is_alert
        
        # Process first face
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region for eyes (upper half of face, with some margin)
        eye_y_start = y + int(h * 0.15)
        eye_y_end = y + int(h * 0.55)
        eye_x_start = x + int(w * 0.05)
        eye_x_end = x + int(w * 0.95)
        
        roi_gray = gray[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
        roi_color = frame[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
        
        # Detect and classify eyes
        ear, eye_status, eye_positions = self.detect_and_classify_eyes(
            roi_gray, roi_color, h
        )
        
        # Update EAR history
        self.ear_history.append(ear)
        
        # Calculate PERCLOS (percentage of frames with low EAR)
        closed_frames = sum(1 for e in self.ear_history if e < self.EAR_THRESHOLD)
        perclos = closed_frames / len(self.ear_history) if self.ear_history else 0
        
        # Alert logic
        is_closed = eye_status == "CLOSED"
        if is_closed:
            self.alert_frames += 1
            if self.alert_frames > self.CONSEC_FRAMES:
                self.is_alert = True
        else:
            self.alert_frames = max(0, self.alert_frames - 2)
            if self.alert_frames == 0:
                self.is_alert = False
        
        # Calculate drowsiness score
        score = 0
        if perclos > 0.3:
            score += 0.4
        if is_closed:
            score += 0.3
        if ear < 0.15:
            score += 0.3
        
        # Draw metrics
        frame = self.draw_metrics(frame, ear, perclos, eye_status, score, len(eye_positions))
        
        # Add eye region rectangle
        cv2.rectangle(frame, (eye_x_start, eye_y_start), 
                     (eye_x_end, eye_y_end), (255, 255, 0), 1)
        
        return frame, self.is_alert
    
    def draw_metrics(self, frame, ear, perclos, eye_status, score, eye_count):
        """Draw metrics on frame"""
        h, w = frame.shape[:2]
        
        # Main info panel - Dark background with transparency
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 300, 230
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # Title
        cv2.putText(frame, "🚗 DROWSINESS DETECTOR", (panel_x + 15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Status
        status_color = (0, 255, 0) if not self.is_alert else (0, 0, 255)
        status_text = "✅ ALERT" if not self.is_alert else "⚠️ DROWSY"
        cv2.putText(frame, status_text, (panel_x + 15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Eye state with emoji
        if eye_status == "OPEN":
            eye_display = "👁️ OPEN"
            eye_color = (0, 255, 0)
        elif eye_status == "CLOSED":
            eye_display = "😴 CLOSED"
            eye_color = (0, 0, 255)
        else:
            eye_display = "❓ UNKNOWN"
            eye_color = (255, 255, 0)
        
        cv2.putText(frame, eye_display, (panel_x + 180, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 1)
        
        # Metrics
        metrics = [
            ("EAR", f"{ear:.3f}", ear < self.EAR_THRESHOLD),
            ("PERCLOS", f"{perclos:.1%}", perclos > 0.3),
            ("Drowsiness Score", f"{score:.2f}", score > 0.4),
            ("Eyes Detected", f"{eye_count}", False),
            ("Alert Frames", f"{self.alert_frames}/{self.CONSEC_FRAMES}", self.alert_frames > 5)
        ]
        
        y_pos = 100
        for label, value, warning in metrics:
            color = (0, 165, 255) if warning else (255, 255, 255)
            cv2.putText(frame, f"{label}: {value}", (panel_x + 15, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 25
        
        # Status bar for EAR
        bar_x, bar_y = panel_x + 15, y_pos + 10
        bar_w, bar_h = 270, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                     (50, 50, 50), -1)
        
        # Fill bar based on EAR
        fill_w = int((ear / 0.5) * bar_w)
        fill_w = min(max(fill_w, 0), bar_w)
        color = (0, 255, 0) if ear > self.EAR_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), 
                     color, -1)
        
        # Threshold line
        threshold_x = bar_x + int((self.EAR_THRESHOLD / 0.5) * bar_w)
        cv2.line(frame, (threshold_x, bar_y - 2), (threshold_x, bar_y + bar_h + 2), 
                (255, 255, 255), 2)
        
        cv2.putText(frame, f"EAR: {ear:.3f}", (bar_x + 5, bar_y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"TH: {self.EAR_THRESHOLD:.2f}", (threshold_x - 30, bar_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Alert banner (flashing)
        if self.is_alert:
            if int(time.time() * 2) % 2 == 0:
                # Full screen alert
                cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 255), -1)
                cv2.putText(frame, "⚠️ DROWSINESS ALERT! ⚠️", 
                            (w//2 - 230, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
                
                # Bottom warning
                cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 255), -1)
                cv2.putText(frame, "⚠️ PULL OVER AND REST IMMEDIATELY! ⚠️", 
                            (w//2 - 250, h-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (w - 160, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*60)
        print("🚗 DRIVER DROWSINESS DETECTION - IMPROVED VERSION")
        print("="*60)
        print("📸 Detection Features:")
        print("   • Face detection with OpenCV")
        print("   • Eye aspect ratio (EAR) calculation")
        print("   • PERCLOS tracking over 30 frames")
        print("   • Real-time alert system")
        print("📸 Press 'q' to quit")
        print("="*60 + "\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, alert = self.process_frame(frame)
            
            cv2.imshow('Drowsiness Detection - Improved', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 System stopped. Stay safe driver!")

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()