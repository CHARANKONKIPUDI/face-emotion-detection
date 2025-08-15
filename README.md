# Face Emotion Detection

A real-time face emotion detection project using **TensorFlow**, **OpenCV**, and a custom-trained CNN model on the FER2013 dataset.  
It detects **7 emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

---

## 🚀 Features
- Trains a CNN model to recognize 7 basic emotions
- Real-time webcam detection using OpenCV
- Preprocessing with grayscale conversion & normalization
- Uses Haarcascade for face detection

---

## 📂 Project Structure
├── dataset/ # Training & testing images (FER2013 format)
│ ├── train/
│ └── test/
├── train.py # Script to train the CNN model
├── testing.py # Script for live webcam emotion detection
├── emotion_model.keras # Saved trained model
├── requirements.txt # Minimal dependencies
└── README.md


---

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/CHARANKONKIPUDI/face-emotion-detection.git
cd face-emotion-detection

2.  **Create and activate a virtual environment**

python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Mac/Linux


3. **Install dependencies**

pip install -r requirements.txt
