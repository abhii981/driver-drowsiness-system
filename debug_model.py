"""
debug_model.py - Test the model directly
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

print("="*60)
print("DEBUGGING MODEL")
print("="*60)

# Load model
model_path = 'eye_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
else:
    print(f"❌ Model not found: {model_path}")
    exit()

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n📸 Press 's' to test an eye crop, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Manually crop eye regions
        left_eye_x = x + int(w * 0.12)
        left_eye_y = y + int(h * 0.22)
        left_eye_w = int(w * 0.25)
        left_eye_h = int(h * 0.18)
        
        right_eye_x = x + int(w * 0.62)
        right_eye_y = y + int(h * 0.22)
        right_eye_w = int(w * 0.25)
        right_eye_h = int(h * 0.18)
        
        # Draw rectangles
        cv2.rectangle(frame, (left_eye_x, left_eye_y), (left_eye_x+left_eye_w, left_eye_y+left_eye_h), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_x, right_eye_y), (right_eye_x+right_eye_w, right_eye_y+right_eye_h), (0, 255, 0), 2)
        
        # Extract eye images
        left_eye = frame[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w]
        right_eye = frame[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w]
        
        # Show eye crops
        if left_eye.size > 0:
            cv2.imshow('Left Eye', left_eye)
        if right_eye.size > 0:
            cv2.imshow('Right Eye', right_eye)
    
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print("\n" + "="*60)
        print("TESTING EYE CROP")
        print("="*60)
        
        if left_eye.size > 0 and right_eye.size > 0:
            # Test left eye
            test_img = cv2.resize(left_eye, (224, 224))
            test_img = test_img / 255.0
            test_img = np.expand_dims(test_img, axis=0)
            pred = model.predict(test_img, verbose=0)
            left_prob = float(pred[0][0])
            left_closed = left_prob > 0.5
            
            # Test right eye
            test_img2 = cv2.resize(right_eye, (224, 224))
            test_img2 = test_img2 / 255.0
            test_img2 = np.expand_dims(test_img2, axis=0)
            pred2 = model.predict(test_img2, verbose=0)
            right_prob = float(pred2[0][0])
            right_closed = right_prob > 0.5
            
            avg_prob = (left_prob + right_prob) / 2
            avg_closed = (1 if left_closed else 0 + 1 if right_closed else 0) / 2
            
            print(f"Left Eye:  prob={left_prob:.3f} -> {'CLOSED' if left_closed else 'OPEN'}")
            print(f"Right Eye: prob={right_prob:.3f} -> {'CLOSED' if right_closed else 'OPEN'}")
            print(f"Average:   prob={avg_prob:.3f} -> {'CLOSED' if avg_closed > 0.5 else 'OPEN'}")
            print("="*60)
        else:
            print("❌ No eyes detected!")

cap.release()
cv2.destroyAllWindows()