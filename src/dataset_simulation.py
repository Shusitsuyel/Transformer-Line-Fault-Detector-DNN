import os
from matplotlib.patheffects import Normal
import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Number of samples
n_samples = 50000

# Fault types
fault_types = ['LG', 'LL', 'LLG', '3PH', 'Normal']

# Simulate data
def simulate_fault_data(n_samples):
    data = []
    for _ in range(n_samples):
        fault = np.random.choice(fault_types, p=[0.2,0.2,0.2,0.2,0.2])
        Va, Vb, Vc = np.random.normal(1.0, 0.02, 3)
        Ia, Ib, Ic = np.random.normal(0.5, 0.05, 3)
        if fault == 'LG':
            Ia *= 0.1
            Va *= 0.5
        elif fault == 'LL':
            Ia *= 0.6
            Ib *= 0.6
        elif fault == 'LLG':
            Ia *= 0.6
            Ib *= 0.6
            Ic *= 0.1
        elif fault == '3PH':
            Va *= 0.3
            Vb *= 0.3
            Vc *= 0.3
            Ia *= 0.3
            Ib *= 0.3
            Ic *= 0.3
        data.append([Va, Vb, Vc, Ia, Ib, Ic, fault])
    df = pd.DataFrame(data, columns=['Voltage_A','Voltage_B','Voltage_C','Current_A','Current_B','Current_C','Fault_Type'])
    return df

# Generate dataset
df = simulate_fault_data(n_samples)

# Ensure absolute path for saving
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
os.makedirs(data_folder, exist_ok=True)
save_path = os.path.join(data_folder, 'simulated_fault_data.csv')

df.to_csv(save_path, index=False)
print(f"Dataset generated and saved to {save_path}")
