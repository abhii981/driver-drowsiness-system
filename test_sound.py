"""
test_sound.py - Test if alarm.mp3 plays
"""

import pygame
import time

print("🔊 Testing alarm sound...")

try:
    pygame.mixer.init()
    pygame.mixer.music.load('alarm.mp3')
    print("✅ Sound loaded successfully!")
    print("🔊 Playing sound...")
    pygame.mixer.music.play()
    
    # Wait for sound to finish
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    
    print("✅ Sound played successfully!")
except Exception as e:
    print(f"❌ Error playing sound: {e}")
    print("   The visual alert will still work as fallback")