"""
model.py
--------
Contains all machine-learning logic for the project.

Classes / Functions:
  TransformerModel   – wraps RandomForestClassifier (condition prediction)
                       and IsolationForest (anomaly detection)
  plot_data()        – visualises the dataset with fault regions highlighted
  plot_feature_importance() – bar chart of feature importances
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from utils import load_data, condition_emoji

# ── Feature columns used for training ────────────────────────────────────────
FEATURES = ["voltage", "load", "temperature", "frequency"]
TARGET   = "condition"


class TransformerModel:
    """
    Encapsulates:
      • RandomForestClassifier  → predicts Normal / Overload / Fault
      • IsolationForest         → flags anomalous sensor readings
    """

    def __init__(self):
        self.rf_model   = RandomForestClassifier(n_estimators=100, random_state=42)
        self.iso_forest = IsolationForest(contamination=0.15, random_state=42)
        self.le         = LabelEncoder()
        self.is_trained = False
        self.feature_names = FEATURES

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, verbose: bool = True) -> dict:
        """
        Load the dataset, train both models, and return evaluation metrics.
        """
        df = load_data()

        X = df[FEATURES].values
        y_raw = df[TARGET].values
        y = self.le.fit_transform(y_raw)           # encode labels → integers

        # Train/test split (80 / 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── 1. Random Forest ──────────────────────────────────────────────────
        self.rf_model.fit(X_train, y_train)
        y_pred = self.rf_model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=self.le.classes_,
            output_dict=True
        )

        # ── 2. Isolation Forest (trained on ALL data – unsupervised) ─────────
        self.iso_forest.fit(X)

        self.is_trained = True

        if verbose:
            print("\n" + "─" * 60)
            print("  📊  MODEL TRAINING COMPLETE")
            print("─" * 60)
            print(f"  Random Forest Accuracy : {acc * 100:.1f}%")
            print(f"  Training samples       : {len(X_train)}")
            print(f"  Testing  samples       : {len(X_test)}")
            print("\n  Classification Report:")
            print(classification_report(
                y_test, y_pred, target_names=self.le.classes_
            ))
            print("  Isolation Forest trained on full dataset (unsupervised).")
            print("─" * 60 + "\n")

        return {"accuracy": acc, "report": report}

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, voltage: float, load: float,
                temperature: float, frequency: float) -> dict:
        """
        Given four sensor values, return:
          condition  – "Normal" / "Overload" / "Fault"
          anomaly    – True if Isolation Forest flags it as anomalous
          confidence – RF class probabilities (0–100 %)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X = np.array([[voltage, load, temperature, frequency]])

        # ── Random Forest prediction ──────────────────────────────────────────
        pred_idx   = self.rf_model.predict(X)[0]
        condition  = self.le.inverse_transform([pred_idx])[0]
        proba      = self.rf_model.predict_proba(X)[0]
        conf       = {cls: round(p * 100, 1)
                      for cls, p in zip(self.le.classes_, proba)}

        # ── Isolation Forest anomaly flag ─────────────────────────────────────
        iso_pred   = self.iso_forest.predict(X)[0]   # -1 = anomaly, +1 = normal
        is_anomaly = (iso_pred == -1)

        return {
            "condition":  condition,
            "anomaly":    is_anomaly,
            "confidence": conf,
        }

    def print_result(self, result: dict,
                     voltage: float, load: float,
                     temperature: float, frequency: float):
        """Pretty-print the prediction result to the console."""
        cond  = result["condition"]
        emoji = condition_emoji(cond)
        conf  = result["confidence"]

        print("\n" + "═" * 60)
        print("  PREDICTION RESULT")
        print("═" * 60)
        print(f"  Voltage     : {voltage} V")
        print(f"  Load        : {load} %")
        print(f"  Temperature : {temperature} °C")
        print(f"  Frequency   : {frequency} Hz")
        print("─" * 60)
        print(f"  Condition   : {emoji}  {cond}")
        print(f"  Anomaly     : {'🔴 YES – Abnormal reading detected!'
                                 if result['anomaly'] else '🟢 NO – Reading within normal bounds'}")
        print("─" * 60)
        print("  Confidence scores (Random Forest):")
        for cls, pct in conf.items():
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {cls:<10} [{bar}] {pct}%")
        print("═" * 60 + "\n")


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_data(save_path: str = "transformer_plot.png"):
    """
    Scatter plot: Temperature vs Load coloured by condition.
    Shaded regions illustrate typical fault zones.
    """
    df = load_data()

    colour_map = {"Normal": "#2ecc71", "Overload": "#f39c12", "Fault": "#e74c3c"}
    marker_map = {"Normal": "o",       "Overload": "s",       "Fault": "^"}

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # ── Fault region highlight ────────────────────────────────────────────────
    ax.axhspan(0, 25,  alpha=0.08, color="red",    label="_fault_low_load")
    ax.axhspan(80, 100, alpha=0.08, color="orange", label="_overload_zone")
    ax.axvspan(100, 150, alpha=0.08, color="red",   label="_fault_high_temp")
    ax.axvspan(75, 100, alpha=0.06, color="orange", label="_overload_temp")

    for condition, group in df.groupby("condition"):
        ax.scatter(
            group["temperature"], group["load"],
            c=colour_map[condition],
            marker=marker_map[condition],
            label=condition,
            s=60, alpha=0.85, edgecolors="white", linewidths=0.3
        )

    # Reference lines
    ax.axvline(75,  color="#f39c12", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(100, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(80,  color="#f39c12", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Temperature (°C)", color="white", fontsize=12)
    ax.set_ylabel("Load (%)",          color="white", fontsize=12)
    ax.set_title("Transformer: Temperature vs Load — Fault Regions Highlighted",
                 color="white", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    legend_patches = [
        mpatches.Patch(color=colour_map[c], label=c)
        for c in ["Normal", "Overload", "Fault"]
    ]
    ax.legend(handles=legend_patches, facecolor="#0f3460",
              labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈  Plot saved → {save_path}")


def plot_feature_importance(model: TransformerModel,
                             save_path: str = "feature_importance.png"):
    """Bar chart of feature importances from the Random Forest."""
    if not model.is_trained:
        raise RuntimeError("Train the model before plotting feature importance.")

    importances = model.rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names_sorted = [FEATURES[i] for i in indices]

    colours = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.bar(names_sorted, importances[indices],
                  color=colours, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", color="white", fontsize=9)

    ax.set_title("Random Forest — Feature Importances",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("Importance", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Feature-importance chart saved → {save_path}")
