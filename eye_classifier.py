
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2
import os

class EyeClassifier:
    def __init__(self, model_path=None):
        """Initialize the eye classifier"""
        self.model = None
        self.input_shape = (224, 224, 3)
        
        if model_path and os.path.exists(model_path):
            print(f" Loading model from {model_path}")
            self.load_model(model_path)
        else:
            print("🆕 Creating new model")
            self.create_model()
    
    def create_model(self):
        """Create a CNN model for eye classification"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 0 = Closed, 1 = Open
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model created successfully!")
        self.model.summary()
        return model
    
    def train_model(self, train_data, train_labels, val_data, val_labels, epochs=15):
        """Train the model"""
        print("Training model...")
        
        history = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        print("Training complete!")
        return history
    
    def predict_eye_state(self, eye_image):
        """Predict if eye is open or closed"""
        if self.model is None:
            raise ValueError("Model not initialized!")
        
        # Preprocess the image
        if len(eye_image.shape) == 2:
            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size
        eye_image = cv2.resize(eye_image, (224, 224))
        
        # Normalize
        eye_image = eye_image / 255.0
        
        # Add batch dimension
        eye_image = np.expand_dims(eye_image, axis=0)
        
        # Predict
        prediction = self.model.predict(eye_image, verbose=0)
        
        # 0 = Closed, 1 = Open
        is_open = prediction[0][0] > 0.5
        confidence = prediction[0][0] if is_open else 1 - prediction[0][0]
        
        return is_open, confidence
    
    def save_model(self, filepath):
        """Save the model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")