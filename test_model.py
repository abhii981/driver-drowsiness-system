"""
test_model.py - Test the model on a single eye image
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load model
model_path = 'eye_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
else:
    print(f"Model not found: {model_path}")
    exit()

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press 's' to save an eye image for testing, 'q' to quit")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

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
        
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h//2, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3, minSize=(20, 20))
        
        for (ex, ey, ew, eh) in eyes[:2]:
            padding = 15
            ex_start = max(0, ex - padding)
            ey_start = max(0, ey - padding)
            ex_end = min(roi_color.shape[1], ex + ew + padding)
            ey_end = min(roi_color.shape[0], ey + eh + padding)
            
            eye_img = roi_color[ey_start:ey_end, ex_start:ex_end]
            if eye_img.size > 0:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Display the eye crop
                cv2.imshow('Eye Crop', eye_img)
    
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the current eye crop for testing
        if 'eye_img' in locals() and eye_img.size > 0:
            cv2.imwrite('test_eye.jpg', eye_img)
            print("Saved test_eye.jpg")
            
            # Test the model on saved image
            test_img = cv2.imread('test_eye.jpg')
            test_img = cv2.resize(test_img, (224, 224))
            test_img = test_img / 255.0
            test_img = np.expand_dims(test_img, axis=0)
            
            pred = model.predict(test_img, verbose=0)
            prob = float(pred[0][0])
            is_closed = prob > 0.5
            print(f"Prediction: {'CLOSED' if is_closed else 'OPEN'} (confidence: {prob:.3f})")

cap.release()
cv2.destroyAllWindows()