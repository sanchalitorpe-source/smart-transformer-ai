"""
main.py
-------
Entry point for the AI-Based Smart Transformer Fault Prediction System.
Run:   python main.py

Menu options:
  1 – Manual input prediction
  2 – Simulate random sensor data
  3 – Simulate a scenario (Normal / Overload / Fault)
  4 – Show dataset plots
  5 – Exit
"""

import sys
from utils import print_banner, validate_inputs, simulate_sensor
from model import TransformerModel, plot_data, plot_feature_importance


def get_float(prompt: str) -> float:
    """Prompt the user until a valid float is entered."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  ⚠️  Please enter a numeric value.")


def manual_prediction(model: TransformerModel):
    """Ask the user to enter four sensor values and display the prediction."""
    print("\n  ── Enter Sensor Readings ──────────────────────────────")
    voltage     = get_float("  Voltage     (V)  [typical 140–260] : ")
    load        = get_float("  Load        (%)  [0–100]           : ")
    temperature = get_float("  Temperature (°C) [20–150]          : ")
    frequency   = get_float("  Frequency   (Hz) [44–52]           : ")

    try:
        validate_inputs(voltage, load, temperature, frequency)
    except ValueError as e:
        print(e)
        return

    result = model.predict(voltage, load, temperature, frequency)
    model.print_result(result, voltage, load, temperature, frequency)


def simulation_mode(model: TransformerModel):
    """Simulate real-time sensor data for a user-chosen scenario."""
    print("\n  Choose simulation scenario:")
    print("    1. Random     (any condition)")
    print("    2. Normal     (healthy transformer)")
    print("    3. Overload   (high load / high temp)")
    print("    4. Fault      (critically abnormal)")

    choice = input("  Enter choice [1-4]: ").strip()
    scenario_map = {"1": "random", "2": "normal", "3": "overload", "4": "fault"}
    scenario = scenario_map.get(choice, "random")

    n = input("  How many readings to simulate? [default: 5] : ").strip()
    n = int(n) if n.isdigit() else 5

    print(f"\n  ── Simulating {n} '{scenario}' sensor readings ──────────")
    for i in range(1, n + 1):
        s = simulate_sensor(scenario)
        result = model.predict(**s)
        cond   = result["condition"]
        anom   = "🔴 ANOMALY" if result["anomaly"] else "🟢 OK"
        print(
            f"  [{i:02d}] V={s['voltage']:6.1f}V  "
            f"L={s['load']:5.1f}%  "
            f"T={s['temperature']:6.1f}°C  "
            f"F={s['frequency']:5.2f}Hz  "
            f"→  {cond:<10}  {anom}"
        )
    print()


def show_plots(model: TransformerModel):
    """Generate and save both visualisation charts."""
    print("\n  Generating charts … ")
    plot_data("transformer_plot.png")
    plot_feature_importance(model, "feature_importance.png")
    print("  Open 'transformer_plot.png' and 'feature_importance.png' to view.\n")


def main():
    print_banner()

    # ── Train models ──────────────────────────────────────────────────────────
    print("  🔧  Initialising models and training …\n")
    model = TransformerModel()
    model.train(verbose=True)

    # ── Main menu loop ────────────────────────────────────────────────────────
    while True:
        print("  ┌──────────────────────────────────┐")
        print("  │           MAIN MENU              │")
        print("  ├──────────────────────────────────┤")
        print("  │  1. Manual sensor input          │")
        print("  │  2. Simulate random readings     │")
        print("  │  3. Run scenario simulation      │")
        print("  │  4. Show / save plots            │")
        print("  │  5. Exit                         │")
        print("  └──────────────────────────────────┘")
        choice = input("  Enter choice [1-5]: ").strip()

        if choice == "1":
            manual_prediction(model)
        elif choice == "2":
            s = simulate_sensor("random")
            print(f"\n  🎲  Random readings → {s}")
            result = model.predict(**s)
            model.print_result(result, **s)
        elif choice == "3":
            simulation_mode(model)
        elif choice == "4":
            show_plots(model)
        elif choice == "5":
            print("\n  👋  Exiting. Goodbye!\n")
            sys.exit(0)
        else:
            print("  ❌  Invalid choice. Please enter 1–5.\n")


if __name__ == "__main__":
    main()
