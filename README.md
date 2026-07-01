# 🚗 Driver Drowsiness Detection System

Real-time driver drowsiness detection using Deep Learning with a custom web UI.

## 📋 Features

- **Real-time face and eye detection** using OpenCV
- **Deep Learning CNN** for eye state classification (Open/Closed)
- **PERCLOS calculation** (Percentage of Eye Closure)
- **Blink detection** - ignores quick blinks
- **Audio + Visual alerts** when drowsiness detected
- **Custom Web UI** with real-time metrics
- **WebSocket** for video streaming

## 🛠️ Tech Stack

### Backend
- Python 3.10+
- TensorFlow/Keras
- OpenCV
- NumPy

### Frontend
- HTML5, CSS3, JavaScript
- WebSocket

### Server
- Node.js
- Express
- ws

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/driver-drowsiness-system.git
cd driver-drowsiness-system