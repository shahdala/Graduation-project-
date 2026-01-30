import os
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import sys
import os

# ✅ Ensure Python finds the 'training' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.load_dataset import train_generator, val_generator  # ✅ Now this should work
from model import build_custom_cnn


# ✅ Define model path
model_path = "models/custom_liver_cnn"

# ✅ Load previous model if exists
if os.path.exists(model_path):
    print("🔄 Resuming training from saved model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("🆕 Starting training from scratch...")
    model = build_custom_cnn()

# ✅ Reduce Learning Rate if Validation Loss Stops Improving
lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)

# ✅ Stop training early if accuracy stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# ✅ Continue training from last checkpoint
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[lr_schedule, early_stop]
)

# ✅ Save model again
model.save(model_path)
print("✅ Model saved successfully!")

# ✅ Save training history
with open("models/training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✅ Training history saved successfully!")
