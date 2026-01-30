import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from load_dataset import val_generator


# Load trained model
model = tf.keras.models.load_model("models/custom_liver_cnn")

# Load training history
try:
    with open("models/training_history.pkl", "rb") as f:
        history = pickle.load(f)
except FileNotFoundError:
    print("❌ Training history not found. Run `train_model.py` first.")
    exit()

# Evaluate model
loss, acc = model.evaluate(val_generator)
print(f"✅ Validation Accuracy: {acc:.2f}")

# Plot Accuracy & Loss Curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training vs Validation Loss")

plt.show()
