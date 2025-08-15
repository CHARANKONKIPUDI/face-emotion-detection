import cv2
import numpy as np
from keras.models import load_model
import os

# ==== PATHS ====
BASE_DIR = r"C:\Users\banga\OneDrive\Desktop\emotion recognization\dataset"
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
TRAIN_DIR = os.path.join(BASE_DIR, "train")

# Load emotion labels dynamically from dataset
emotion_labels = sorted(os.listdir(TRAIN_DIR))
print(f"[INFO] Emotion labels: {emotion_labels}")

# Load trained model
model = load_model(MODEL_PATH)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.expand_dims(face, axis=-1)  # channel dimension
        face = np.expand_dims(face, axis=0)   # batch dimension

        predictions = model.predict(face, verbose=0)
        emotion_idx = np.argmax(predictions)
        emotion_text = emotion_labels[emotion_idx]

        # Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Live Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
