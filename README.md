# 🚗 Driver Drowsiness Detection System

Real-time driver drowsiness detection using a fine-tuned CNN model with a web-based interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-green.svg)
![Node.js](https://img.shields.io/badge/Node.js-18+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📌 Quick Links

- **Dataset Used:** [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset) (85,000+ images, **too large to include in this repo**)
- **Live Demo:** Coming soon
- **Report Issues:** [GitHub Issues](https://github.com/abhii981/driver-drowsiness-system/issues)

---

## 🎯 What This Project Does

This system monitors a driver's eyes in real-time using a webcam. When it detects signs of drowsiness (like prolonged eye closure), it triggers both visual and audio alerts to warn the driver.

The core idea is simple: **if your eyes stay closed for more than a couple of seconds, you're probably too tired to drive.**

---

## ✨ Key Features

- **Real-time eye tracking** using your webcam
- **Deep learning model** (82.45% accuracy) to tell if eyes are open or closed
- **PERCLOS scoring** (percentage of time eyes are closed) as a fatigue metric
- **Blink detection** that ignores quick blinks and only alerts on sustained closure
- **Audio + visual alerts** when drowsiness is detected
- **Web-based dashboard** showing live metrics (EAR, PERCLOS, etc.)
- **Adjustable sensitivity** settings
- **WebSocket streaming** for low-latency video

---

## 🛠️ How It's Built

### Backend
- Python with TensorFlow/Keras for the deep learning model
- OpenCV for face and eye detection
- Node.js with WebSocket for real-time communication

### Frontend
- Plain HTML, CSS, and JavaScript
- Live metrics dashboard
- Responsive design

---

## 📂 Project Structure

Here's what the codebase looks like:

```
driver-drowsiness-system/
├── public/              # Web interface files
│   ├── audio/alarm.mp3  # Alert sound
│   ├── css/style.css
│   ├── js/main.js
│   └── index.html
├── detector_ws.py       # Main detection logic
├── eye_classifier.py    # Model loading/prediction
├── train_small.py       # Training script
├── server.js            # Node.js server
├── requirements.txt     # Python dependencies
└── package.json         # Node dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A webcam

### Installation Steps

**1. Clone the repository:**
```bash
git clone https://github.com/abhii981/driver-drowsiness-system.git
cd driver-drowsiness-system
```

**2. Set up a Python virtual environment:**
```bash
python -m venv tf_env
# On Windows:
tf_env\Scripts\activate
# On Mac/Linux:
source tf_env/bin/activate
```

**3. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**4. Install Node.js dependencies:**
```bash
npm install
```

**5. (Optional) Train the model:**
```bash
python train_small.py
```
*Note: This uses a small sample dataset. For full training, download the [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset).*

**6. Start the application:**
```bash
node server.js
```

**7. Open your browser and go to:**
```
http://localhost:3000
```

---

## 🔬 How Detection Works

1. **Face detection** – OpenCV locates your face in the camera feed
2. **Eye extraction** – The system crops out just the eye regions
3. **Eye state classification** – A CNN predicts if each eye is open or closed
4. **PERCLOS calculation** – Tracks the percentage of closed-eye frames over time
5. **Blink handling** – Quick blinks (<0.5 seconds) are ignored
6. **Alert trigger** – Sustained eye closure triggers a warning

---

## 🧪 Testing Guide

| What you do | What you should see |
|-------------|---------------------|
| Keep eyes open | "OPEN" (green) – no alert |
| Quick blink | "BLINKING" (orange) – no alert |
| Close eyes for 2+ seconds | "SLEEPY" (red) – alert triggers! |

### Helper Scripts
- `python test_camera.py` – Check if your webcam works
- `python test_sound.py` – Test the alarm sound

---

## 📊 Model & Dataset

### Dataset
- **Source:** [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)
- **Size:** 85,000+ images of eyes (open and closed states)
- **Note:** This dataset is **too large to include directly** in this repository. Download it from Kaggle if you want to retrain the model.

### Model Performance
- **Validation accuracy:** 82.45%
- **Best recorded validation accuracy:** 86.55%
- **Training data:** 2,000 sample images (subset)

---

## 🧠 Why I Built This

I noticed that most driver fatigue systems are either expensive or rely on external hardware. This project is my attempt to build an accessible, webcam-based alternative using open-source tools.

### Challenges & Lessons Learned

The biggest challenge was balancing accuracy with performance. Running a deep learning model on a CPU while maintaining 30 FPS video streaming required careful optimization, including reducing the model size and using WebSocket for efficient data transfer.

---

## 🔜 Future Improvements

- Train on the full MRL dataset for higher accuracy
- Add mobile app support
- Integrate with smartwatch heart rate data
- Implement facial recognition for driver identification
- Add drowsiness history logging

---

## 🤝 Contributing

Found a bug or have an idea? Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Abhinandan**
- GitHub: [@abhii981](https://github.com/abhii981)
- Project Link: [driver-drowsiness-system](https://github.com/abhii981/driver-drowsiness-system)

---

## 🙏 Credits

- [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset) for the training data
- TensorFlow, OpenCV, and MediaPipe for making computer vision accessible

---

## ⭐ Support This Project

If you found this useful, please consider **starring** the repository on GitHub. It helps others discover the project!

---

**Made with 💻 and ☕ for safer roads** 🚗