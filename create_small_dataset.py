"""
create_small_dataset.py - Create a small dataset from the full dataset
Takes only 1000 images from each class for fast training
"""

import os
import shutil
import random
from pathlib import Path

print("📂 Creating small dataset for testing...")

# Paths
SOURCE = "data"  # Your full dataset
DEST = "data_small"  # Small dataset

# Number of images per class
SAMPLE_SIZE = 1000  # 1000 awake + 1000 sleepy = 2000 total

# Create destination structure
for split in ['train', 'val', 'test']:
    for category in ['awake', 'sleepy']:
        dest_path = os.path.join(DEST, split, category)
        os.makedirs(dest_path, exist_ok=True)

# Copy random samples
for split in ['train', 'val', 'test']:
    for category in ['awake', 'sleepy']:
        src_path = os.path.join(SOURCE, split, category)
        dest_path = os.path.join(DEST, split, category)
        
        if not os.path.exists(src_path):
            continue
            
        # Get all images
        images = [f for f in os.listdir(src_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Take random sample
        sample = random.sample(images, min(SAMPLE_SIZE, len(images)))
        
        # Copy images
        for img in sample:
            shutil.copy2(os.path.join(src_path, img), 
                        os.path.join(dest_path, img))
        
        print(f"   {split}/{category}: {len(sample)} images copied")

print("\n✅ Small dataset created in 'data_small' folder!")
print(f"   Total images: ~{SAMPLE_SIZE * 2 * 3} (train/val/test)")