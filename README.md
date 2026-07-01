# 🚗 Driver Drowsiness Detection System

> Real-time driver drowsiness detection using Deep Learning (CNN) with a custom web UI and WebSocket streaming.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-green.svg)
![Node.js](https://img.shields.io/badge/Node.js-18+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Model Training](#model-training)
- [Results & Accuracy](#results--accuracy)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## 🎯 Overview

This project detects driver drowsiness in real-time using a **Deep Learning CNN model** trained on eye images. It monitors eye state (Open/Closed), calculates PERCLOS (Percentage of Eye Closure), and triggers audio-visual alerts when drowsiness is detected.

The system uses:
- **OpenCV** for face and eye detection
- **TensorFlow/Keras** for eye state classification
- **WebSocket** for real-time video streaming
- **Custom HTML/CSS/JS UI** for monitoring

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Real-time Detection** | Processes webcam feed at 30 FPS |
| 🧠 **Deep Learning Model** | CNN with 82.45% validation accuracy |
| 👁️ **Eye State Classification** | Detects Open/Closed eyes |
| 📊 **PERCLOS Tracking** | Monitors percentage of eye closure |
| 😉 **Blink Detection** | Ignores quick blinks (< 0.5 seconds) |
| 🔊 **Audio Alert** | Plays alarm sound when drowsy |
| 🚨 **Visual Alert** | Red banner with flashing effect |
| 🖥️ **Custom Web UI** | Real-time metrics dashboard |
| ⚙️ **Adjustable Settings** | EAR threshold and sensitivity controls |
| 📈 **Live Metrics** | EAR, PERCLOS, Drowsiness Score |

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**
- **TensorFlow/Keras** - Deep Learning
- **OpenCV** - Face and eye detection
- **NumPy** - Numerical operations

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with gradients and animations
- **JavaScript** - Real-time updates
- **WebSocket** - Video streaming

### Server
- **Node.js** - HTTP server
- **Express** - Static file serving
- **ws** - WebSocket server

---

## 📂 Project Structure

```
driver-drowsiness-system/
│
├── public/                          # Frontend files
│   ├── audio/
│   │   └── alarm.mp3               # Alert sound
│   ├── css/
│   │   └── style.css               # UI styling
│   ├── js/
│   │   └── main.js                 # Frontend logic
│   └── index.html                  # Main UI page
│
├── detector_ws.py                   # Main detection engine
├── eye_classifier.py                # DL model class
├── train_small.py                   # Model training script
├── server.js                        # Node.js server
├── requirements.txt                 # Python dependencies
├── package.json                     # Node dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Documentation
```

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/abhii981/driver-drowsiness-system.git
cd driver-drowsiness-system
```

### 2️⃣ Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv tf_env

# Activate (Windows)
tf_env\Scripts\activate

# Activate (Mac/Linux)
source tf_env/bin/activate
```

### 3️⃣ Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Install Node.js Dependencies
```bash
npm install
```

### 5️⃣ Train the Model (Optional)
If you want to train your own model on a dataset:
```bash
python train_small.py
```

### 6️⃣ Run the Application
```bash
node server.js
```

### 7️⃣ Open Browser
```
http://localhost:3000
```

---

## 🔬 How It Works

### Detailed Flow:

1. **Face Detection** → OpenCV Haar Cascade detects face
2. **Eye Extraction** → Extracts eye regions from face
3. **Deep Learning** → CNN classifies eyes as Open/Closed
4. **PERCLOS Tracking** → Tracks percentage of eye closure over 30 frames
5. **Blink Logic** → Ignores quick blinks (threshold: 15 frames ≈ 0.5-0.7 seconds)
6. **Alert Trigger** → Visual banner + audio alarm when drowsy

---

## 🧪 Testing

### To test the system:

| Action | Expected Result |
|--------|-----------------|
| **Open eyes normally** | Shows "👁️ OPEN" (Green) → No alert |
| **Quick blink** | Shows "😉 BLINKING" (Orange) → No alert |
| **Close eyes for 2+ seconds** | Shows "😴 SLEEPY" (Red) → Alert triggers! |

### Test Sound:
```bash
python test_sound.py
```

### Test Camera:
```bash
python test_camera.py
```

---

## 📊 Model Training

### Dataset: MRL Eye Dataset
- **85,000+** images of human eyes
- **Train/Val/Test** split: 60/20/20
- **Classes**: Awake (Open) and Sleepy (Closed)

### Training Results

| Dataset | Accuracy |
|---------|----------|
| **Small Dataset (2,000 images)** | **82.45%** |
| **Best Validation** | **86.55%** |

### Training Command
```bash
python train_small.py
```

### Data Preprocessing (Optional)
```bash
python preprocess_data.py
```

---

## 📸 Screenshots

### UI Dashboard
![UI Dashboard](screenshots/dashboard.png)

### Alert Triggered
![Alert](screenshots/alert.png)

### Metrics Panel
![Metrics](screenshots/metrics.png)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

**Abhishek**

- GitHub: [@abhii981](https://github.com/abhii981)
- Project Link: [https://github.com/abhii981/driver-drowsiness-system](https://github.com/abhii981/driver-drowsiness-system)

---

## 🙏 Acknowledgments

- [MRL Eye Dataset](https://www.kaggle.com/datasets/kayvanshah/eye-dataset) - Dataset used for training
- [TensorFlow](https://www.tensorflow.org/) - Deep Learning framework
- [OpenCV](https://opencv.org/) - Computer Vision library

---

## ⭐ Star the Project!

If you found this helpful, please give it a ⭐ on GitHub!

---

**Made with ❤️ for safer roads 🚗**
