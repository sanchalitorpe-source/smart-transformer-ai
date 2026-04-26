"""
utils.py
--------
Helper / utility functions used across the project.
• load_data()         – reads the CSV and returns a DataFrame
• print_banner()      – prints a decorative project title
• validate_inputs()   – checks that user-entered values are in realistic ranges
• simulate_sensor()   – returns a random set of sensor readings for simulation mode
"""

import pandas as pd
import numpy as np
import os

# ── File path ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "transformer_data.csv")


def load_data() -> pd.DataFrame:
    """Load the transformer dataset from CSV."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Run:  python data/generate_dataset.py   to create it first."
        )
    df = pd.read_csv(DATA_PATH)
    return df


def print_banner():
    """Print a decorative ASCII banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║   ⚡  AI-BASED SMART TRANSFORMER FAULT PREDICTION  ⚡   ║
║          AIML Mini Project  |  Python + sklearn          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


# ── Realistic sensor ranges ───────────────────────────────────────────────────
SENSOR_LIMITS = {
    "voltage":     (140.0, 260.0),
    "load":        (0.0,   100.0),
    "temperature": (20.0,  150.0),
    "frequency":   (44.0,   52.0),
}


def validate_inputs(voltage: float, load: float,
                    temperature: float, frequency: float) -> bool:
    """
    Check that all four sensor readings lie within physically plausible limits.
    Returns True if valid, raises ValueError with a helpful message if not.
    """
    values = {
        "voltage": voltage,
        "load": load,
        "temperature": temperature,
        "frequency": frequency,
    }
    for name, val in values.items():
        lo, hi = SENSOR_LIMITS[name]
        if not (lo <= val <= hi):
            raise ValueError(
                f"  ❌  '{name}' value {val} is out of realistic range "
                f"[{lo} – {hi}]."
            )
    return True


def simulate_sensor(scenario: str = "random") -> dict:
    """
    Return a dict of simulated sensor values.

    scenario options:
        'random'   – completely random (may be any class)
        'normal'   – typical healthy transformer readings
        'overload' – high load / high temperature readings
        'fault'    – critically abnormal readings
    """
    rng = np.random.default_rng()

    if scenario == "normal":
        return {
            "voltage":     round(rng.uniform(210, 240), 2),
            "load":        round(rng.uniform(20, 60), 2),
            "temperature": round(rng.uniform(30, 65), 2),
            "frequency":   round(rng.uniform(49.5, 50.5), 2),
        }
    elif scenario == "overload":
        return {
            "voltage":     round(rng.uniform(200, 215), 2),
            "load":        round(rng.uniform(80, 100), 2),
            "temperature": round(rng.uniform(75, 100), 2),
            "frequency":   round(rng.uniform(48.0, 49.4), 2),
        }
    elif scenario == "fault":
        return {
            "voltage":     round(rng.uniform(160, 195), 2),
            "load":        round(rng.uniform(0, 20), 2),
            "temperature": round(rng.uniform(100, 140), 2),
            "frequency":   round(rng.uniform(45.0, 47.9), 2),
        }
    else:  # random
        return {
            "voltage":     round(rng.uniform(140, 260), 2),
            "load":        round(rng.uniform(0, 100), 2),
            "temperature": round(rng.uniform(20, 150), 2),
            "frequency":   round(rng.uniform(44, 52), 2),
        }


def condition_emoji(condition: str) -> str:
    """Return a status emoji for a given condition label."""
    return {"Normal": "✅", "Overload": "⚠️", "Fault": "🚨"}.get(condition, "❓")
