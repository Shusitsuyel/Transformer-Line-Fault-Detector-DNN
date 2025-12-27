from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# ---------------------------
# Load model, scaler, label encoder
# ---------------------------
model_path = os.path.join(os.path.dirname(__file__), 'models', 'dnn_fault_model.h5')
model = load_model(model_path)

# Load scaler and label encoder
with open(os.path.join('models', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join('models', 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)

# ---------------------------
# Routes
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            Va = float(request.form['Va'])
            Vb = float(request.form['Vb'])
            Vc = float(request.form['Vc'])
            Ia = float(request.form['Ia'])
            Ib = float(request.form['Ib'])
            Ic = float(request.form['Ic'])

            # Prepare input
            sample = np.array([[Va, Vb, Vc, Ia, Ib, Ic]])
            sample_scaled = scaler.transform(sample)
            pred = model.predict(sample_scaled)
            pred_label = le.inverse_transform([np.argmax(pred)])[0]

            return render_template('result.html', fault_type=pred_label)

        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
