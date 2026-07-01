"""
detector_ws.py - With Confidence-Based Detection
"""

import cv2
import numpy as np
import sys
import json
import base64
from collections import deque
import time
import os
from tensorflow.keras.models import load_model

class EyeClassifier:
    def __init__(self, model_path='eye_model.h5'):
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        if os.path.exists(model_path):
            print(json.dumps({'type': 'log', 'level': 'success', 'message': f'✅ Loading model from {model_path}'}), flush=True)
            self.model = load_model(model_path)
            print(json.dumps({'type': 'log', 'level': 'success', 'message': '✅ Model loaded!'}), flush=True)
        else:
            print(json.dumps({'type': 'log', 'level': 'warning', 'message': f'⚠️ Model not found'}), flush=True)
            self.model = None
    
    def predict(self, eye_image):
        if self.model is None:
            return None, 0.0
        
        try:
            if len(eye_image.shape) == 2:
                eye_image = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2RGB)
            eye_image = cv2.resize(eye_image, (224, 224))
            eye_image = eye_image / 255.0
            eye_image = np.expand_dims(eye_image, axis=0)
            
            prediction = self.model.predict(eye_image, verbose=0)
            prob = float(prediction[0][0])
            
            # prob > 0.5 = sleepy (closed), prob < 0.5 = awake (open)
            is_closed = prob > 0.5
            confidence = prob if is_closed else 1 - prob
            
            return is_closed, confidence, prob
        except Exception as e:
            return None, 0.0, 0.0

class Detector:
    def __init__(self):
        print(json.dumps({'type': 'log', 'level': 'info', 'message': '🚗 Initializing...'}), flush=True)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        self.eye_classifier = EyeClassifier('eye_model.h5')
        self.use_dl = self.eye_classifier.model is not None
        
        # Confidence threshold - only use DL if confidence > this
        self.CONFIDENCE_THRESHOLD = 0.65
        
        self.eye_history = deque(maxlen=30)
        self.is_alert = False
        self.alert_frames = 0
        self.running = False
        self.cap = None
        self.frame_count = 0
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print(json.dumps({'type': 'log', 'level': 'error', 'message': '❌ Could not open webcam!'}), flush=True)
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(json.dumps({'type': 'log', 'level': 'success', 'message': '✅ Webcam started'}), flush=True)
        print(json.dumps({'type': 'log', 'level': 'success', 'message': '✅ System ready!'}), flush=True)
    
    def extract_eyes(self, frame, face_roi):
        x, y, w, h = face_roi
        
        eye_y_start = y + int(h * 0.15)
        eye_y_end = y + int(h * 0.55)
        eye_x_start = x + int(w * 0.05)
        eye_x_end = x + int(w * 0.95)
        
        roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
        roi_color = frame[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.05, 3, minSize=(20, 20))
        
        eye_images = []
        eye_positions = []
        
        for (ex, ey, ew, eh) in eyes[:2]:
            padding = 10
            ex_start = max(0, ex - padding)
            ey_start = max(0, ey - padding)
            ex_end = min(roi_color.shape[1], ex + ew + padding)
            ey_end = min(roi_color.shape[0], ey + eh + padding)
            
            eye_img = roi_color[ey_start:ey_end, ex_start:ex_end]
            
            if eye_img.size > 0 and eye_img.shape[0] > 10 and eye_img.shape[1] > 10:
                eye_images.append(eye_img)
                eye_positions.append((ex + eye_x_start, ey + eye_y_start, ew, eh))
        
        return eye_images, eye_positions, roi_color
    
    def process_frame(self):
        if not self.running:
            return None, None, False
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None, False
        
        self.frame_count += 1
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        metrics = {
            'ear': 0.25,
            'perclos': 0.0,
            'score': 0.0,
            'is_closed': False,
            'is_yawn': False,
            'head_nod': False,
            'blink_rate': 0,
            'use_dl': self.use_dl
        }
        
        if len(faces) == 0:
            cv2.putText(frame, "No face", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return frame, metrics, False
        
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        eye_images, eye_positions, roi_color = self.extract_eyes(frame, (x, y, w, h))
        
        is_closed = False
        ear = 0.25
        dl_confidence = 0.0
        dl_enabled = False
        
        # Try DL prediction
        if self.use_dl and len(eye_images) >= 2:
            predictions = []
            confidences = []
            probs = []
            
            for eye_img in eye_images[:2]:
                pred_closed, conf, prob = self.eye_classifier.predict(eye_img)
                if pred_closed is not None:
                    predictions.append(pred_closed)
                    confidences.append(conf)
                    probs.append(prob)
            
            if predictions:
                # Average the predictions
                avg_closed = sum(predictions) / len(predictions)
                avg_confidence = np.mean(confidences)
                avg_prob = np.mean(probs)
                
                # Only use DL if confidence is high enough
                if avg_confidence > self.CONFIDENCE_THRESHOLD:
                    is_closed = avg_closed > 0.5
                    dl_confidence = avg_confidence
                    dl_enabled = True
                    
                    # EAR-like metric from probability
                    ear = avg_prob * 0.5 + 0.1
                    
                    cv2.putText(frame, f"DL: {'CLOSED' if is_closed else 'OPEN'} ({avg_confidence:.2f})", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                (0, 0, 255) if is_closed else (0, 255, 0), 1)
                else:
                    cv2.putText(frame, f"DL: Low conf ({avg_confidence:.2f})", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Fallback to EAR if DL not confident or not available
        if not dl_enabled and len(eye_positions) >= 2:
            eyes_sorted = sorted(eye_positions, key=lambda e: e[0])
            left_ear = eyes_sorted[0][3] / (eyes_sorted[0][2] + 1)
            right_ear = eyes_sorted[1][3] / (eyes_sorted[1][2] + 1)
            ear = (left_ear + right_ear) / 2
            ear = min(max(ear * (h / 200) * 2, 0.05), 0.45)
            is_closed = ear < 0.20
            
            cv2.putText(frame, "Using EAR (DL low conf)", (10, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Draw eye rectangles
        for (ex, ey, ew, eh) in eye_positions:
            color = (0, 0, 255) if is_closed else (0, 255, 0)
            cv2.rectangle(roi_color, (ex - 10, ey - 10), (ex+ew+10, ey+eh+10), color, 2)
        
        # Update history
        self.eye_history.append(1 if is_closed else 0)
        perclos = float(sum(self.eye_history) / len(self.eye_history)) if self.eye_history else 0.0
        
        # Alert logic
        if is_closed:
            self.alert_frames += 1
            if self.alert_frames > 15:
                self.is_alert = True
        else:
            self.alert_frames = max(0, self.alert_frames - 1)
            if self.alert_frames == 0:
                self.is_alert = False
        
        # Score
        score = 0.0
        if perclos > 0.3:
            score += 0.5
        if self.is_alert:
            score += 0.3
        if ear < 0.15:
            score += 0.2
        
        # Display
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"PERCLOS: {perclos:.2%}", (10, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"DL: {'ON' if self.use_dl else 'OFF'}", (10, 66), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # State
        if self.is_alert:
            state_text = "😴 SLEEPY"
            text_color = (0, 0, 255)
        elif is_closed:
            state_text = "😉 BLINKING"
            text_color = (0, 165, 255)
        else:
            state_text = "👁️ OPEN"
            text_color = (0, 255, 0)
        
        cv2.putText(frame, state_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        if self.is_alert:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (frame.shape[1]//2 - 100, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        metrics = {
            'ear': float(ear),
            'perclos': float(perclos),
            'score': float(score),
            'is_closed': bool(is_closed),
            'is_yawn': False,
            'head_nod': False,
            'blink_rate': 0,
            'use_dl': self.use_dl
        }
        
        return frame, metrics, self.is_alert

def main():
    detector = Detector()
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.05)
                continue
            
            data = json.loads(line.strip())
            
            if data.get('type') == 'start_detection':
                detector.running = True
                print(json.dumps({'type': 'log', 'level': 'success', 'message': '🟢 Detection started'}), flush=True)
                
                while detector.running:
                    frame, metrics, alert = detector.process_frame()
                    
                    if frame is not None:
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        print(json.dumps({'type': 'video_frame', 'frame': frame_b64}), flush=True)
                        print(json.dumps({'type': 'metrics', 'metrics': metrics}), flush=True)
                        
                        if alert:
                            print(json.dumps({'type': 'alert', 'status': True}), flush=True)
                    
                    time.sleep(0.03)
                
            elif data.get('type') == 'stop_detection':
                detector.running = False
                detector.is_alert = False
                detector.alert_frames = 0
                print(json.dumps({'type': 'log', 'level': 'info', 'message': '🔴 Detection stopped'}), flush=True)
                
            elif data.get('type') == 'exit':
                break
                
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(json.dumps({'type': 'log', 'level': 'error', 'message': f'Error: {e}'}), flush=True)

if __name__ == "__main__":
    main()