# ⚡ AI-Based Smart Transformer Fault Prediction System
### AIML Mini Project | Python + scikit-learn

---

## 📁 Project Structure

```
transformer_fault_prediction/
│
├── main.py                  ← CLI entry point (run this first)
├── model.py                 ← ML models: RandomForest + IsolationForest
├── utils.py                 ← Helper functions (load data, validate, simulate)
├── gui.py                   ← Optional Tkinter GUI
├── requirements.txt         ← Python library dependencies
│
└── data/
    ├── generate_dataset.py  ← Script to (re)create the CSV dataset
    └── transformer_data.csv ← 200-row synthetic dataset (auto-generated)
```

---

## 📋 File-by-File Explanation

| File | Purpose |
|------|---------|
| `main.py` | The CLI menu: trains models, accepts user input, runs simulations |
| `model.py` | Defines `TransformerModel` class — trains RF + ISO Forest, predicts, plots |
| `utils.py` | `load_data()`, `validate_inputs()`, `simulate_sensor()`, banner printer |
| `gui.py` | Tkinter window with input boxes, Predict/Simulate/Clear buttons, confidence bars |
| `data/generate_dataset.py` | Generates 200 synthetic rows (Normal / Overload / Fault) |
| `data/transformer_data.csv` | The dataset used for training |

---

## 📊 Dataset Details

| Feature | Normal Range | Overload Range | Fault Range | Unit |
|---------|-------------|----------------|-------------|------|
| `voltage` | 210 – 240 | 200 – 215 | 160 – 195 | Volts |
| `load` | 20 – 60 | 80 – 100 | 0 – 20 | % |
| `temperature` | 30 – 65 | 75 – 100 | 100 – 140 | °C |
| `frequency` | 49.5 – 50.5 | 48.0 – 49.4 | 45.0 – 47.9 | Hz |

**Label distribution:** Normal: 70 | Overload: 70 | Fault: 60 | **Total: 200 rows**

---

## 🤖 How the Models Work (Simple Explanation)

### 1. Random Forest Classifier
- Think of it as **100 decision trees voting** on the outcome.
- Each tree asks questions like: *"Is temperature > 75? Is load > 80%?"*
- All trees vote, and the majority wins → **Normal / Overload / Fault**
- Gives a confidence score (e.g. 85% Fault, 15% Overload)

### 2. Isolation Forest (Anomaly Detection)
- Does **NOT** use labels — it's unsupervised.
- Builds random trees that try to *isolate* each data point.
- **Anomalous** points are isolated quickly (fewer splits needed).
- **Normal** points take many splits to isolate.
- Output: **+1** (normal) or **-1** (anomaly)

---

## ⚙️ Setup & Installation

### Step 1 — Install Python libraries
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install pandas scikit-learn matplotlib numpy
```

### Step 2 — Generate dataset (if CSV is missing)
```bash
python data/generate_dataset.py
```

### Step 3 — Run CLI version
```bash
python main.py
```

### Step 4 — Run GUI version (optional)
```bash
python gui.py
```

---

## 🖥️ CLI Menu Options

```
1. Manual sensor input      → Enter voltage, load, temp, frequency manually
2. Simulate random readings → Generates one random reading and predicts
3. Run scenario simulation  → Simulate N readings from Normal/Overload/Fault
4. Show / save plots        → Saves PNG charts to project folder
5. Exit
```

---

## 📈 Output Plots

After running option 4 (or the test script), two PNG files are saved:

| File | Description |
|------|-------------|
| `transformer_plot.png` | Scatter: Temperature vs Load, coloured by condition |
| `feature_importance.png` | Bar chart: which sensor matters most to the RF model |

---

## 🎯 Example Predictions

| Voltage | Load | Temp | Freq | Expected | Result |
|---------|------|------|------|----------|--------|
| 225 V | 40% | 55°C | 50.0 Hz | ✅ Normal | Normal |
| 205 V | 90% | 85°C | 48.5 Hz | ⚠️ Overload | Overload |
| 178 V | 8% | 125°C | 46.2 Hz | 🚨 Fault | Fault |

---

## 🧪 Quick Test (No Menu)

```python
from model import TransformerModel

m = TransformerModel()
m.train()

result = m.predict(voltage=225, load=40, temperature=55, frequency=50.0)
print(result)
# → {'condition': 'Normal', 'anomaly': False, 'confidence': {'Fault': 0.0, 'Normal': 100.0, 'Overload': 0.0}}
```

---

## 🎤 Viva Questions & Answers

### Q1. What is the purpose of this project?
**A:** This project simulates an AI-powered monitoring system for electrical transformers. It uses machine learning to analyse real-time sensor data (voltage, load, temperature, frequency) and automatically predict whether the transformer is in a Normal, Overload, or Fault condition — without human inspection.

---

### Q2. Why did you choose Random Forest for this problem?
**A:** Random Forest is ideal here because:
- It handles small datasets well (only 200 rows)
- It's robust to outliers and noisy sensor readings
- It provides feature importance scores (which sensor matters most)
- It gives class probabilities, not just a single label
- It doesn't need feature scaling (unlike SVM or KNN)
- It avoids overfitting better than a single Decision Tree

---

### Q3. What is Isolation Forest and why use it alongside Random Forest?
**A:** Isolation Forest is an **unsupervised anomaly detection** algorithm. While Random Forest needs labelled training data, Isolation Forest works without labels. It isolates data points by randomly selecting a feature and a split value. Anomalous points (unusual sensor readings) get isolated in fewer steps. We use it as a **second layer of safety** — if a reading is classified as "Normal" by RF but is still statistically unusual, the Isolation Forest can flag it.

---

### Q4. What do the four sensor features represent physically?
**A:**
- **Voltage** — The electrical potential delivered by the transformer. A drop signals load problems or a fault.
- **Load (%)** — How much of the transformer's rated capacity is being used. Above 80% risks overheating.
- **Temperature (°C)** — Core/winding temperature. Above 75°C is concerning; above 100°C indicates fault.
- **Frequency (Hz)** — Grid frequency. Should stay close to 50 Hz (India/EU). Deviation indicates grid instability.

---

### Q5. What is the difference between supervised and unsupervised learning? How does your project use both?
**A:**
- **Supervised learning** uses labelled data — you tell the model what the correct answer is during training. → *Random Forest Classifier* uses labelled rows (Normal/Overload/Fault).
- **Unsupervised learning** finds patterns without labels. → *Isolation Forest* only sees sensor values, no labels. It learns what "normal" looks like statistically.

This dual approach gives better reliability: RF predicts the class, Isolation Forest detects novel anomalies not covered by training labels.

---

### Q6. How would you improve this project for real-world deployment?
**A:**
- **Real sensor data**: Connect to IoT sensors via MQTT or REST API
- **Time series**: Use LSTM or sliding-window features for temporal patterns
- **Online learning**: Update the model incrementally as new data arrives
- **Alerting**: Send SMS/email alerts when Fault is predicted
- **Dashboard**: Replace Tkinter with a web dashboard (Flask + React)
- **More features**: Add oil level, humidity, harmonic distortion
- **Model persistence**: Save trained model with `joblib` so retraining isn't needed every run

---

### Q7. What is feature importance and what did your model learn?
**A:** Feature importance tells us how much each input feature contributed to the Random Forest's decisions. In this project, **temperature** and **voltage** are typically the most important features because:
- Temperature rises significantly in both Overload and Fault conditions
- Voltage drops sharply during faults (short circuits)
- Load distinguishes Overload (very high %) from Fault (near zero — short circuit)
- Frequency changes are secondary indicators

This matches real-world electrical engineering knowledge, which validates our synthetic dataset design.

---

## 📦 Dependencies

```
pandas        – data loading and manipulation
scikit-learn  – RandomForestClassifier, IsolationForest, train_test_split
matplotlib    – plotting / visualisation
numpy         – numerical operations
tkinter       – GUI (built into Python standard library)
```

---

*Made with ❤️ for AIML Engineering Lab | Python 3.8+*
