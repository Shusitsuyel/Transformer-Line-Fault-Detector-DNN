import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path=None):
    """
    Load and preprocess dataset for DNN.
    """
    # If no path provided, use default simulated_fault_data.csv
    if file_path is None:
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
        file_path = os.path.join(data_folder, 'simulated_fault_data.csv')

    df = pd.read_csv(file_path)

    # Features: voltage & current
    X = df[['Voltage_A','Voltage_B','Voltage_C','Current_A','Current_B','Current_C']].values
    y = df['Fault_Type'].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, le, scaler


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = load_and_preprocess_data()
    print("Data loaded and preprocessed successfully.")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
