"""
drowsiness_detector.py - Driver Drowsiness Detection with Deep Learning
Compatible with TensorFlow 2.21.0
"""

import cv2
import numpy as np
from collections import deque
import time
import threading  # ← ADD THIS IMPORT
from eye_classifier import EyeClassifier
import os

class DrowsinessDetector:
    def __init__(self, use_dl=True):
        print("🚗 Initializing Drowsiness Detector with Deep Learning...")
        
        # Load OpenCV face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize Deep Learning Eye Classifier
        self.use_dl = use_dl
        self.eye_classifier = None
        
        if use_dl:
            print("🧠 Loading Deep Learning model...")
            try:
                # Check if model exists, if not create one
                if os.path.exists('eye_model.h5'):
                    self.eye_classifier = EyeClassifier('eye_model.h5')
                else:
                    self.eye_classifier = EyeClassifier()
                    # Save the model for future use
                    self.eye_classifier.save_model('eye_model.h5')
                print("✅ Deep Learning model ready")
            except Exception as e:
                print(f"⚠️ Error loading DL model: {e}")
                print("🔄 Falling back to OpenCV only")
                self.use_dl = False
        
        # Thresholds
        self.EAR_THRESHOLD = 0.20
        self.CONSEC_FRAMES = 15
        
        # History tracking
        self.eye_history = deque(maxlen=30)
        self.alert_frames = 0
        self.is_alert = False
        self.alarm_playing = False  # ← ADD THIS
        
        print("✅ Detector initialized!")
    
    def extract_eye_region(self, frame, face_roi):
        """Extract eye regions for classification"""
        x, y, w, h = face_roi
        
        # Upper half of face for eyes
        roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h//2, x:x+w]
        
        # Detect eyes
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3, minSize=(15, 15))
        
        eye_images = []
        eye_positions = []
        
        for (ex, ey, ew, eh) in eyes:
            # Extract eye region
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            if eye_img.size > 0:
                eye_images.append(eye_img)
                eye_positions.append((ex, ey, ew, eh))
        
        return eye_images, eye_positions, roi_color
    
    def play_alarm(self):
        """Play the alarm sound"""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load('alarm.mp3')
            pygame.mixer.music.play()
            # Keep playing until manually stopped
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            # Fallback: just print
            print(f"🔊 ALARM! Drowsiness detected! (Sound error: {e})")
        finally:
            self.alarm_playing = False  # ← Reset when done
    
    def process_frame(self, frame):
        """Process a single frame with Deep Learning"""
        if frame is None or frame.size == 0:
            return frame, False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            # Display message on frame
            cv2.putText(frame, "No face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame, self.is_alert
        
        # Get first face
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract and classify eyes
        eye_images, eye_positions, roi_color = self.extract_eye_region(frame, (x, y, w, h))
        
        is_closed = False
        ear = 0.25  # Default EAR
        
        if len(eye_images) >= 2 and self.use_dl and self.eye_classifier:
            # Use Deep Learning for eye classification
            open_eyes = 0
            confidences = []
            
            for eye_img in eye_images[:2]:  # Take first two eyes
                try:
                    is_open, confidence = self.eye_classifier.predict_eye_state(eye_img)
                    if is_open:
                        open_eyes += 1
                    confidences.append(confidence)
                except Exception as e:
                    print(f"DL prediction error: {e}")
                    continue
            
            if confidences:
                # Determine if eyes are closed
                is_closed = open_eyes < 1  # Both eyes closed = drowsy
                avg_confidence = np.mean(confidences)
                
                # Calculate EAR-like metric from confidence
                ear = avg_confidence * 0.4 + 0.1
                
                # Show DL prediction on frame
                status = "OPEN" if not is_closed else "CLOSED"
                cv2.putText(frame, f"DL: {status} ({avg_confidence:.2f})", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        elif len(eye_positions) >= 2:
            # Fallback to EAR calculation
            eyes_sorted = sorted(eye_positions, key=lambda e: e[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[1]
            
            # Calculate EAR
            left_ear = left_eye[3] / (left_eye[2] + 1)
            right_ear = right_eye[3] / (right_eye[2] + 1)
            ear = (left_ear + right_ear) / 2
            ear = min(max(ear * (h / 200) * 2, 0.05), 0.45)
            
            is_closed = ear < self.EAR_THRESHOLD
        
        # Draw eye rectangles
        for (ex, ey, ew, eh) in eye_positions:
            color = (0, 0, 255) if is_closed else (0, 255, 0)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
        
        # Update history
        self.eye_history.append(1 if is_closed else 0)
        perclos = sum(self.eye_history) / len(self.eye_history) if self.eye_history else 0
        
        # Alert logic
        if is_closed:
            self.alert_frames += 1
            if self.alert_frames > self.CONSEC_FRAMES:
                self.is_alert = True
        else:
            self.alert_frames = max(0, self.alert_frames - 1)
            if self.alert_frames == 0:
                self.is_alert = False
        
        # Play alarm when alert is triggered
        if self.is_alert and not self.alarm_playing:
            self.alarm_playing = True
            threading.Thread(target=self.play_alarm, daemon=True).start()
        
        # Calculate drowsiness score
        score = 0
        if perclos > 0.3:
            score += 0.5
        if is_closed:
            score += 0.3
        if ear < 0.15:
            score += 0.2
        
        # Draw metrics
        frame = self.draw_metrics(frame, ear, perclos, is_closed, score, self.is_alert)
        
        return frame, self.is_alert
    
    def draw_metrics(self, frame, ear, perclos, is_closed, score, alert):
        """Draw metrics on frame"""
        h, w = frame.shape[:2]
        
        # Info panel
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 280, 240
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Title
        cv2.putText(frame, "🚗 DROWSINESS DETECTOR", (panel_x + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Status
        status_color = (0, 255, 0) if not alert else (0, 0, 255)
        status_text = "✅ ALERT" if not alert else "⚠️ DROWSY"
        cv2.putText(frame, status_text, (panel_x + 10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Eye state
        eye_emoji = "👁️" if not is_closed else "😴"
        eye_text = "OPEN" if not is_closed else "CLOSED"
        cv2.putText(frame, f"{eye_emoji} {eye_text}", (panel_x + 160, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Metrics
        metrics = [
            f"EAR: {ear:.3f}",
            f"PERCLOS: {perclos:.2%}",
            f"Score: {score:.2f}",
            f"Alert Frames: {self.alert_frames}/{self.CONSEC_FRAMES}",
            f"DL: {'ON' if self.use_dl else 'OFF'}"
        ]
        
        y_pos = 95
        for metric in metrics:
            color = (255, 255, 255)
            if "EAR" in metric and ear < self.EAR_THRESHOLD:
                color = (0, 165, 255)
            if "PERCLOS" in metric and perclos > 0.3:
                color = (0, 165, 255)
            if "Score" in metric and score > 0.4:
                color = (0, 165, 255)
            cv2.putText(frame, metric, (panel_x + 10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 22
        
        # Alert banner
        if alert:
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 255), -1)
                cv2.putText(frame, "⚠️ DROWSINESS ALERT! ⚠️", 
                            (w//2 - 200, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                cv2.putText(frame, "⚠️ PULL OVER AND REST! ⚠️", 
                            (w//2 - 180, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (w - 150, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        print("🚗 DRIVER DROWSINESS DETECTION WITH DEEP LEARNING")
        print("="*60)
        print(f"🧠 Deep Learning: {'ON' if self.use_dl else 'OFF'}")
        print("📸 Press 'q' to quit")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, alert = self.process_frame(frame)
            
            cv2.imshow('Drowsiness Detection with DL', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 System stopped. Stay safe driver!")

if __name__ == "__main__":
    detector = DrowsinessDetector(use_dl=False)
    detector.run()