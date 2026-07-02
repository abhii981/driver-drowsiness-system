"""
train_small.py - Transfer Learning with MobileNetV2
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
import os
import time
import numpy as np

print("="*70)
print("TRAINING WITH TRANSFER LEARNING (MobileNetV2)")
print("="*70)

DATA_PATH = "data_better"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

if not os.path.exists(DATA_PATH):
    print(f"Dataset not found: {DATA_PATH}")
    exit()

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['awake', 'sleepy'],
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_PATH, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['awake', 'sleepy'],
    shuffle=False
)

print(f"\nTraining: {train_generator.samples} images")
print(f"Validation: {val_generator.samples} images")

# Class weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\nClass Weights:")
print(f"  Awake (class 0): {class_weight_dict[0]:.2f}")
print(f"  Sleepy (class 1): {class_weight_dict[1]:.2f}")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'eye_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("\nStarting training...")
print("="*70)

start_time = time.time()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/60:.1f} minutes")

val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy: {val_acc:.2%}")

model.save('eye_model.h5')
print("Model saved as 'eye_model.h5'")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nNext steps:")
print("  1. Run: node server.js")
print("  2. Open: http://localhost:3000")
print("  3. Click Start Detection")