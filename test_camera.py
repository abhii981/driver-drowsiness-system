import cv2

print("📸 Testing webcam...")

# Try to open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam!")
    print("Please check:")
    print("1. Camera is connected")
    print("2. No other app is using the camera")
    exit()
else:
    print("✅ Webcam opened successfully!")

# Capture and display
ret, frame = cap.read()
if ret:
    print(f"✅ Frame captured! Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print("📺 Showing webcam feed... Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Webcam Test - Press ESC', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
else:
    print("❌ Could not capture frame!")

cap.release()
cv2.destroyAllWindows()
print("👋 Test complete!")