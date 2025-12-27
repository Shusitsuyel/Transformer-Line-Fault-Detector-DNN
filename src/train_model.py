import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import load_and_preprocess_data

# --------------------------
# 1. Load preprocessed data
# --------------------------
X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = load_and_preprocess_data()
print("Data loaded successfully.")
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

# --------------------------
# 2. Define the DNN model
# --------------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --------------------------
# 3. Ensure models folder exists
# --------------------------
model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
os.makedirs(model_folder, exist_ok=True)
model_path = os.path.join(model_folder, 'dnn_fault_model.h5')

# --------------------------
# 4. Callbacks
# --------------------------
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# --------------------------
# 5. Train the model
# --------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[checkpoint, early_stop]
)

# --------------------------
# 6. Evaluate the model
# --------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# --------------------------
# 7. Save training plot
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid(True)
plot_path = os.path.join(model_folder, 'training_accuracy.png')
plt.savefig(plot_path)
plt.show()

# --------------------------
# 8. Print confirmation
# --------------------------
print(f"\nModel saved at: {model_path}")
print(f"Training accuracy plot saved at: {plot_path}")
