import os
import pickle
from data_preprocessing import load_and_preprocess_data

# Load dataset and get scaler and label encoder
X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = load_and_preprocess_data()

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Scaler and Label Encoder saved successfully in 'models/' folder.")
