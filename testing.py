import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model("emotion_model.keras")

# Class labels (order should match your training folders)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load and preprocess image
img_path = "test_image.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=-1)  # add channel
img = np.expand_dims(img, axis=0)   # add batch

# Predict
pred = model.predict(img)
emotion = class_labels[np.argmax(pred)]
print(f"Predicted Emotion: {emotion}")
