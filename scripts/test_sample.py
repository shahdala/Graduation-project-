import os
import cv2
import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------
# 1) Configuration
# --------------------------------------------------------------------
MODEL_PATH = "models/custom_liver_cnn"   # Path to your trained Keras model
IMAGE_FOLDER = "ttt"                     # Folder with images to classify
IMG_SIZE = (224, 224)                    # The size your model expects

# Suppose your model outputs [Benign, Malignant, Normal]
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

# Define a confidence threshold for Benign or Malignant
THRESHOLD = 0.5

# --------------------------------------------------------------------
# 2) Load Model
# --------------------------------------------------------------------
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# --------------------------------------------------------------------
# 3) Iterate Over Images, Resize to 224×224, Classify
# --------------------------------------------------------------------
for filename in os.listdir(IMAGE_FOLDER):
    img_path = os.path.join(IMAGE_FOLDER, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"❌ Could not load image: {img_path}")
        continue

    # Resize to the expected size
    image = cv2.resize(image, IMG_SIZE)

    # Expand dims: (1, 224, 224, 3)
    image_batch = np.expand_dims(image, axis=0) / 255.0

    # ----------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------
    prediction = model.predict(image_batch)
    scores = prediction[0]   # e.g. [score_benign, score_malignant, score_normal]

    benign_score = scores[0]
    malignant_score = scores[1]
    normal_score = scores[2]

    # ----------------------------------------------------------------
    # Custom Logic:
    # If neither benign nor malignant score is above THRESHOLD,
    # we classify as "Normal" by default.
    # Otherwise, pick whichever is highest among the 3.
    # ----------------------------------------------------------------
    if (benign_score < THRESHOLD) and (malignant_score < THRESHOLD):
        # Force "Normal" if neither is confident enough
        predicted_class = "Normal"
    else:
        # Otherwise pick the highest
        predicted_index = np.argmax(scores)
        predicted_class = CLASS_NAMES[predicted_index]

    # ----------------------------------------------------------------
    # Print or Store Results
    # ----------------------------------------------------------------
    print(f"🔍 File: {filename}")
    print(f"   Scores: {scores}")
    print(f"   Predicted Class: {predicted_class}")
    print("-" * 50)
