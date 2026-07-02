
import os
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm

print("="*70)
print("📂 CREATING DATASET (LOWER QUALITY STANDARDS)")
print("="*70)

# Configuration
SOURCE = "data"  # Full dataset
DEST = "data_better"  # Output
IMAGES_PER_CLASS = 3000  # ← Increased to 3000 per class

# LOWER STANDARDS (keeps more images)
MIN_BRIGHTNESS = 15    # Was 30 (now accepts darker images)
BLUR_THRESHOLD = 30    # Was 50 (now accepts blurrier images)

def check_image_quality(image):
    """Check if image is good quality - LOWER STANDARDS"""
    brightness = np.mean(image)
    if brightness < MIN_BRIGHTNESS:
        return False, "dark"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return False, "blurry"
    
    return True, "good"

def enhance_image(image):
    """Enhance image quality"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Less aggressive
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)  # Less denoising
    return enhanced

# Clear existing
if os.path.exists(DEST):
    shutil.rmtree(DEST)

# Create structure
for split in ['train', 'val']:
    for category in ['awake', 'sleepy']:
        os.makedirs(os.path.join(DEST, split, category), exist_ok=True)

print("\n📊 Processing images...")
print("-"*40)

stats = {'total': 0, 'valid': 0, 'invalid': 0, 'dark': 0, 'blurry': 0}

for split in ['train', 'val']:
    for category in ['awake', 'sleepy']:
        src_path = os.path.join(SOURCE, split, category)
        dst_path = os.path.join(DEST, split, category)
        
        if not os.path.exists(src_path):
            print(f"   ⚠️ {split}/{category} not found")
            continue
        
        # Get all images
        images = [f for f in os.listdir(src_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n   Processing {split}/{category}: {len(images)} images")
        
        # Take more images (3000 per class)
        sample_size = min(IMAGES_PER_CLASS, len(images))
        selected = random.sample(images, sample_size)
        
        valid_count = 0
        for img_file in tqdm(selected):
            stats['total'] += 1
            img_path = os.path.join(src_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                stats['invalid'] += 1
                continue
            
            is_good, reason = check_image_quality(img)
            if not is_good:
                stats['invalid'] += 1
                if reason == 'dark':
                    stats['dark'] += 1
                elif reason == 'blurry':
                    stats['blurry'] += 1
                continue
            
            # Enhance and save
            enhanced = enhance_image(img)
            enhanced = cv2.resize(enhanced, (224, 224))
            dst_file = os.path.join(dst_path, img_file)
            cv2.imwrite(dst_file, enhanced)
            valid_count += 1
            stats['valid'] += 1
        
        print(f"      Valid: {valid_count}, Invalid: {sample_size - valid_count}")

print("\n" + "-"*70)
print(" Dataset Statistics (LOWER STANDARDS):")
print(f"   Total processed: {stats['total']}")
print(f"   Valid (kept): {stats['valid']} ✅")
print(f"   Invalid (removed): {stats['invalid']}")
print(f"   Dark: {stats['dark']}")
print(f"   Blurry: {stats['blurry']}")
print(f"\n Dataset saved to: {DEST}")
print("="*70)