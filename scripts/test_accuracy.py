import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------------
# 1) Define Paths & Parameters
# -------------------------------------------------------------------
# The "dataset_augmented" folder has three subfolders: benign, normal, malignant.
DATASET_DIR = "dataset_augmented"  # e.g., "dataset_augmented"
MODEL_PATH = r"C:\Users\pc\Desktop\taron\Liver_model_taron\models\custom_liver_cnn"  # Path to your model

# If your model expects a certain input size (e.g. 224x224, 380x380, etc.)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Adjust as needed

# -------------------------------------------------------------------
# 2) Create Data Generator for the Entire Folder
# -------------------------------------------------------------------
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # Important for correct label alignment
)

# -------------------------------------------------------------------
# 3) Load the Pretrained Model
# -------------------------------------------------------------------
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------------------------------------------
# 4) Evaluate (Loss & Accuracy)
# -------------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"✅ Evaluation on '{DATASET_DIR}'")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_acc:.4f}")

# -------------------------------------------------------------------
# 5) Classification Report & Confusion Matrix
# -------------------------------------------------------------------
# Get predictions
y_prob = model.predict(test_generator, verbose=1)  # Probabilities
y_pred = np.argmax(y_prob, axis=1)                 # Class indices

# True labels from the generator
y_true = test_generator.classes

# Map label indices to class names
class_indices = test_generator.class_indices
idx_to_class = dict((v, k) for k, v in class_indices.items())
class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

# Print Classification Report
print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Print Confusion Matrix
print("✅ Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
