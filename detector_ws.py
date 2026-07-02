"""
detector_ws.py - DL Primary with EAR Fallback (CAMERA FIXED)
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

print(json.dumps({'type': 'log', 'level': 'info', 'message': 'Starting DL Primary detector...'}), flush=True)

# Load DL Model
model = None
model_path = 'eye_model.h5'
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        print(json.dumps({'type': 'log', 'level': 'success', 'message': 'Model loaded'}), flush=True)
    except Exception as e:
        print(json.dumps({'type': 'log', 'level': 'warning', 'message': f'Model load error: {e}'}), flush=True)
        # Don't exit, continue without model
        model = None

# Open camera - FIXED: removed sys.exit on failure
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(json.dumps({'type': 'log', 'level': 'error', 'message': 'Could not open webcam'}), flush=True)
    # Try alternative camera index
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print(json.dumps({'type': 'log', 'level': 'error', 'message': 'Could not open webcam with index 1 either'}), flush=True)
        # Keep trying with index 0 as fallback
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(json.dumps({'type': 'log', 'level': 'error', 'message': 'FATAL: Could not open any webcam'}), flush=True)
            sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# State
running = False
is_alert = False
alert_frames = 0
eye_history = deque(maxlen=30)
CONSEC_FRAMES = 12

# Smoothing
smooth_state = False
smooth_count = 0
SMOOTH_THRESHOLD = 3

print(json.dumps({'type': 'log', 'level': 'success', 'message': 'System ready'}), flush=True)

def predict_eye_dl(eye_img):
    """DL prediction - PRIMARY"""
    if model is None:
        return None, 0.0
    try:
        if len(eye_img.shape) == 2:
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)
        eye_img = cv2.resize(eye_img, (224, 224))
        eye_img = eye_img / 255.0
        eye_img = np.expand_dims(eye_img, axis=0)
        pred = model.predict(eye_img, verbose=0)
        prob = float(pred[0][0])
        # prob > 0.5 = CLOSED
        is_closed = prob > 0.5
        return is_closed, prob
    except:
        return None, 0.0

def calculate_ear(eyes, face_h):
    """EAR - FALLBACK"""
    if len(eyes) < 2:
        return 0.25, False
    
    eyes_sorted = sorted(eyes, key=lambda e: e[0])
    eye_areas = [(ex, ey, ew, eh, ew*eh) for (ex, ey, ew, eh) in eyes_sorted]
    eye_areas_sorted = sorted(eye_areas, key=lambda e: e[4], reverse=True)
    
    if len(eye_areas_sorted) < 2:
        return 0.25, False
    
    left_eye = eye_areas_sorted[0][:4]
    right_eye = eye_areas_sorted[1][:4]
    
    left_ear = left_eye[3] / (left_eye[2] + 1)
    right_ear = right_eye[3] / (right_eye[2] + 1)
    ear = (left_ear + right_ear) / 2
    ear = min(max(ear * (face_h / 200) * 2, 0.05), 0.45)
    is_closed = ear < 0.20
    
    return ear, is_closed

while True:
    try:
        line = sys.stdin.readline()
        if not line:
            time.sleep(0.05)
            continue
        
        data = json.loads(line.strip())
        
        if data.get('type') == 'start_detection':
            running = True
            alert_frames = 0
            is_alert = False
            smooth_state = False
            smooth_count = 0
            print(json.dumps({'type': 'log', 'level': 'success', 'message': 'Detection started'}), flush=True)
            
            while running:
                ret, frame = cap.read()
                if not ret:
                    print(json.dumps({'type': 'log', 'level': 'warning', 'message': 'Frame read failed'}), flush=True)
                    time.sleep(0.1)
                    continue
                
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                # Initialize default metrics
                metrics = {
                    'ear': 0.25,
                    'perclos': 0.0,
                    'score': 0.0,
                    'is_closed': False,
                    'is_yawn': False,
                    'head_nod': False,
                    'blink_rate': 0,
                    'use_dl': model is not None
                }
                
                if len(faces) == 0:
                    cv2.putText(frame, "No face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    metrics['is_closed'] = False
                    print(json.dumps({'type': 'video_frame', 'frame': frame_b64}), flush=True)
                    print(json.dumps({'type': 'metrics', 'metrics': metrics}), flush=True)
                    time.sleep(0.03)
                    continue
                
                # Get face
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Detect eyes
                roi_gray = gray[y:y+h//2, x:x+w]
                roi_color = frame[y:y+h//2, x:x+w]
                
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3, minSize=(20, 20))
                
                eye_positions = []
                eye_images = []
                
                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_positions.append((ex + x, ey + y, ew, eh))
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Extract eye image for DL
                    eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                    if eye_img.size > 0:
                        eye_images.append(eye_img)
                
                # Initialize variables
                ear = 0.25
                is_closed = False
                
                # ============================================
                # PRIMARY: DL PREDICTION
                # ============================================
                dl_is_closed = None
                dl_prob = 0.0
                dl_confidence = 0.0
                
                if model is not None and len(eye_images) >= 2:
                    dl_results = []
                    for eye_img in eye_images[:2]:
                        closed, prob = predict_eye_dl(eye_img)
                        if closed is not None:
                            dl_results.append((closed, prob))
                    
                    if dl_results:
                        avg_closed = sum(1 for c, _ in dl_results if c) / len(dl_results)
                        dl_prob = sum(p for _, p in dl_results) / len(dl_results)
                        dl_confidence = dl_prob if avg_closed > 0.5 else 1 - dl_prob
                        dl_is_closed = avg_closed > 0.5
                        
                        # Show DL on camera
                        dl_status = "CLOSED" if dl_is_closed else "OPEN"
                        color = (0, 0, 255) if dl_is_closed else (0, 255, 0)
                        cv2.putText(frame, f"DL: {dl_status} ({dl_prob:.3f})", 
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # ============================================
                # DECISION: DL if confident, else EAR
                # ============================================
                if dl_is_closed is not None and dl_confidence > 0.6:
                    # DL is confident - use it
                    is_closed = dl_is_closed
                    cv2.putText(frame, "Using DL (confident)", (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    # DL not confident - use EAR fallback
                    ear, ear_closed = calculate_ear(eye_positions, h)
                    is_closed = ear_closed
                    cv2.putText(frame, f"Using EAR: {ear:.3f}", (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # ============================================
                # SMOOTHING
                # ============================================
                if is_closed:
                    smooth_count += 1
                else:
                    smooth_count -= 1
                
                smooth_count = max(0, min(smooth_count, SMOOTH_THRESHOLD + 1))
                
                if smooth_count >= SMOOTH_THRESHOLD:
                    smooth_state = True
                elif smooth_count <= 0:
                    smooth_state = False
                
                is_closed = smooth_state
                
                # ============================================
                # ALERT LOGIC
                # ============================================
                eye_history.append(1 if is_closed else 0)
                perclos = float(sum(eye_history) / len(eye_history)) if eye_history else 0.0
                
                if is_closed:
                    alert_frames += 1
                    if alert_frames >= CONSEC_FRAMES:
                        is_alert = True
                else:
                    alert_frames = 0
                    is_alert = False
                
                # Score
                score = 0.0
                if perclos > 0.3:
                    score += 0.5
                if is_alert:
                    score += 0.3
                
                # ============================================
                # DISPLAY
                # ============================================
                cv2.putText(frame, f"PERCLOS: {perclos:.2%}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {alert_frames}/{CONSEC_FRAMES}", (10, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"State: {'CLOSED' if is_closed else 'OPEN'}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_closed else (0, 255, 0), 2)
                
                # Face label
                if is_alert:
                    state_text = "SLEEPY - ALERT!"
                    text_color = (0, 0, 255)
                elif is_closed:
                    state_text = "BLINKING"
                    text_color = (0, 165, 255)
                else:
                    state_text = "AWAKE"
                    text_color = (0, 255, 0)
                
                cv2.putText(frame, state_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                # Alert banner
                if is_alert:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 255), -1)
                    cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (frame.shape[1]//2 - 130, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                    cv2.putText(frame, "PULL OVER!", (frame.shape[1]//2 - 70, frame.shape[0] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # ============================================
                # SEND TO UI
                # ============================================
                metrics = {
                    'ear': float(ear),
                    'perclos': float(perclos),
                    'score': float(score),
                    'is_closed': bool(is_closed),
                    'is_yawn': False,
                    'head_nod': False,
                    'blink_rate': 0,
                    'use_dl': model is not None
                }
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                print(json.dumps({'type': 'video_frame', 'frame': frame_b64}), flush=True)
                print(json.dumps({'type': 'metrics', 'metrics': metrics}), flush=True)
                
                if is_alert:
                    print(json.dumps({'type': 'alert', 'status': True}), flush=True)
                
                time.sleep(0.03)
        
        elif data.get('type') == 'stop_detection':
            running = False
            is_alert = False
            alert_frames = 0
            smooth_count = 0
            smooth_state = False
            print(json.dumps({'type': 'log', 'level': 'info', 'message': 'Detection stopped'}), flush=True)
        
        elif data.get('type') == 'exit':
            break
            
    except json.JSONDecodeError:
        continue
    except Exception as e:
        print(json.dumps({'type': 'log', 'level': 'error', 'message': f'Error: {e}'}), flush=True)

cap.release()
print(json.dumps({'type': 'log', 'level': 'info', 'message': 'Shutting down'}), flush=True)