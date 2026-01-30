import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  # ✅ Use OpenCV to check if images load properly

# Define dataset path
dataset_path = "dataset_augmented/"
img_size = (224, 224)
batch_size = 32

# Create Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # ✅ 80% Train, 20% Validation
)

# Load Training Data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

# Load Validation Data
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# ✅ Print Class Labels to Verify Correct Labeling
print("Class Labels Mapping:", train_generator.class_indices)

# ✅ Show Sample Images to Verify Labels
x_batch, y_batch = train_generator.next()
plt.figure(figsize=(10,5))

for i in range(6):
    img_path = train_generator.filepaths[i]  # ✅ Get file path
    print(f"Checking image: {img_path}")  # ✅ Print file path

    # ✅ Load image manually to check visibility
    img = cv2.imread(img_path)  # Read image
    if img is None:
        print(f"❌ ERROR: Image {img_path} could not be loaded!")
    else:
        print(f"✅ Image {img_path} loaded successfully.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ✅ Convert to RGB
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"Label: {np.argmax(y_batch[i])}")
    plt.axis("off")

plt.show()
