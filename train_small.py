"""
train_small.py - Fast training on small dataset
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time

print("🚀 Training on SMALL dataset...")
print("="*60)

DATA_PATH = "data_small"  # Small dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['awake', 'sleepy']
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_PATH, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['awake', 'sleepy']
)

print(f"\n✅ Training: {train_generator.samples} images")
print(f"✅ Validation: {val_generator.samples} images")

# Smaller model for faster training
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n🧠 Model Summary:")
model.summary()

print("\n🚀 Starting training...")
print("⏱️ Estimated time: 10-15 minutes")
print("="*60)

start_time = time.time()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")

# Evaluate
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"📊 Validation Accuracy: {val_acc:.2%}")

# Save model
model.save('eye_model.h5')
print("✅ Model saved as 'eye_model.h5'")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE! Model ready.")
print("="*60)
print("\n📝 Next steps:")
print("   1. Run: node server.js")
print("   2. Open: http://localhost:3000")
print("   3. Click Start Detection")