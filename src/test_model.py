import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import load_and_preprocess_data

# Load trained model
model = load_model('../models/dnn_fault_model.h5')

# Load preprocessed test data
X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = load_and_preprocess_data()

## Example: Enter your own values
Va = float(input("Enter Voltage A: "))
Vb = float(input("Enter Voltage B: "))
Vc = float(input("Enter Voltage C: "))
Ia = float(input("Enter Current A: "))
Ib = float(input("Enter Current B: "))
Ic = float(input("Enter Current C: "))

# Prepare input for model
sample = np.array([[Va, Vb, Vc, Ia, Ib, Ic]])
sample_scaled = scaler.transform(sample)  # scale same as training
pred = model.predict(sample_scaled)
pred_label = le.inverse_transform([np.argmax(pred)])[0]

print(f"Predicted Fault Type: {pred_label}")