# ⚡ Transmission Line Fault Detection using DNN

A Deep Neural Network (DNN) based system for classification of transmission line faults
using voltage and current signals, with a professional Flask web interface.

---

## 🔍 Fault Types Detected
- Normal
- LG (Line to Ground)
- LL (Line to Line)
- LLG (Double Line to Ground)
- LLL (Three Phase Fault)

---

## 🧠 Technologies Used
- Python 3.13
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Flask (Web UI)
- HTML + CSS (Professional UI)

---

## 📂 Project Structure
fault detection/
├── src/
├── UI/
├── data/
├── requirements.txt
├── README.md
└── .gitignore


---

## 🚀 How to Run

1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Train the model

python src/train_model.py

3️⃣ Save preprocessors

python src/save_preprocessors.py

4️⃣ Run Flask UI

python UI/app.py

Open browser:

http://127.0.0.1:5000


📊 Input Parameters

> Va, Vb, Vc (Voltages)

> Ia, Ib, Ic (Currents)

> Frequency

🧪 Dataset

Simulated transmission line fault dataset generated using MATLAB-style signals.

👨‍💻 Author

Shushil Suyel
B.Tech CSE (AI)

📜 License

This project is for academic and research purposes.


---

5️⃣ Initialize Git Locally

Open terminal **inside project root**:

```bash
git init
git status

6️⃣ Commit Your Project

git add .
git commit -m "Initial commit: DNN-based transmission line fault detection with Flask UI"

