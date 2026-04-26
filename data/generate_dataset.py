"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset for transformer fault prediction.
Run this once to create transformer_data.csv in the data/ folder.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_data(n_samples=200):
    records = []

    # ── NORMAL (70 samples) ──────────────────────────────────────────────────
    n_normal = 70
    voltage_n     = np.random.uniform(210, 240, n_normal)   # Volts (ideal: 220-230)
    load_n        = np.random.uniform(20, 60, n_normal)      # % load
    temperature_n = np.random.uniform(30, 65, n_normal)      # °C  (safe < 75)
    frequency_n   = np.random.uniform(49.5, 50.5, n_normal)  # Hz  (ideal: 50)
    for i in range(n_normal):
        records.append([
            round(voltage_n[i], 2),
            round(load_n[i], 2),
            round(temperature_n[i], 2),
            round(frequency_n[i], 2),
            "Normal"
        ])

    # ── OVERLOAD (70 samples) ────────────────────────────────────────────────
    n_overload = 70
    voltage_o     = np.random.uniform(200, 215, n_overload)  # slight drop
    load_o        = np.random.uniform(80, 100, n_overload)   # very high load
    temperature_o = np.random.uniform(75, 100, n_overload)   # high temp
    frequency_o   = np.random.uniform(48.0, 49.4, n_overload)# slight drop
    for i in range(n_overload):
        records.append([
            round(voltage_o[i], 2),
            round(load_o[i], 2),
            round(temperature_o[i], 2),
            round(frequency_o[i], 2),
            "Overload"
        ])

    # ── FAULT (60 samples) ───────────────────────────────────────────────────
    n_fault = 60
    voltage_f     = np.random.uniform(160, 195, n_fault)    # severe drop
    load_f        = np.random.uniform(0, 20, n_fault)        # near zero (short circuit)
    temperature_f = np.random.uniform(100, 140, n_fault)     # dangerously high
    frequency_f   = np.random.uniform(45.0, 47.9, n_fault)  # severe drop
    for i in range(n_fault):
        records.append([
            round(voltage_f[i], 2),
            round(load_f[i], 2),
            round(temperature_f[i], 2),
            round(frequency_f[i], 2),
            "Fault"
        ])

    df = pd.DataFrame(records, columns=["voltage", "load", "temperature", "frequency", "condition"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate_data()
    out_path = os.path.join(os.path.dirname(__file__), "transformer_data.csv")
    df.to_csv(out_path, index=False)
    print(f"✅  Dataset saved → {out_path}  ({len(df)} rows)")
    print(df["condition"].value_counts())
