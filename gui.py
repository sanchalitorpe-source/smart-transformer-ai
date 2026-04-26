"""
gui.py
------
Optional Tkinter GUI for the Transformer Fault Prediction System.
Run:   python gui.py

Features:
  • Input fields  – Voltage, Load, Temperature, Frequency
  • Predict button – calls the trained model
  • Clear button   – resets all fields
  • Simulate button– fills fields with random sensor data
  • Result panel   – shows condition + anomaly + confidence bars
  • Status-colour  – Green (Normal) | Orange (Overload) | Red (Fault)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

from utils import validate_inputs, simulate_sensor, condition_emoji
from model import TransformerModel

# ── Theme colours ─────────────────────────────────────────────────────────────
BG       = "#1a1a2e"
PANEL    = "#16213e"
ACCENT   = "#0f3460"
WHITE    = "#e0e0e0"
GREEN    = "#2ecc71"
ORANGE   = "#f39c12"
RED      = "#e74c3c"
BLUE     = "#3498db"
FONT     = ("Segoe UI", 11)
FONT_B   = ("Segoe UI", 11, "bold")
FONT_H   = ("Segoe UI", 14, "bold")


class TransformerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⚡ AI Transformer Fault Prediction System")
        self.geometry("620x680")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.model    = TransformerModel()
        self.trained  = False
        self._build_ui()
        self._train_in_background()

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=ACCENT, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⚡  Smart Transformer Fault Predictor",
                 bg=ACCENT, fg=WHITE, font=("Segoe UI", 15, "bold")).pack()
        tk.Label(hdr, text="AI / ML Mini Project  —  Random Forest + Isolation Forest",
                 bg=ACCENT, fg="#aaa", font=("Segoe UI", 9)).pack()

        # Status strip
        self.status_var = tk.StringVar(value="🔧  Training model … please wait")
        self.status_lbl = tk.Label(self, textvariable=self.status_var,
                                   bg=ORANGE, fg="black", font=FONT_B, pady=5)
        self.status_lbl.pack(fill="x")

        # ── Input section ─────────────────────────────────────────────────────
        inp_frame = tk.LabelFrame(self, text=" 📡  Sensor Input ",
                                  bg=PANEL, fg=WHITE, font=FONT_B,
                                  padx=20, pady=12, bd=2, relief="groove")
        inp_frame.pack(fill="x", padx=20, pady=(14, 6))

        fields = [
            ("Voltage (V)",     "voltage_var",     "Typical: 210 – 240 V"),
            ("Load (%)",        "load_var",        "Typical: 20 – 60 %"),
            ("Temperature (°C)","temperature_var", "Typical: 30 – 65 °C"),
            ("Frequency (Hz)",  "frequency_var",   "Typical: 49.5 – 50.5 Hz"),
        ]
        self.vars = {}
        for i, (label, var_name, hint) in enumerate(fields):
            tk.Label(inp_frame, text=label, bg=PANEL, fg=WHITE,
                     font=FONT, width=16, anchor="w").grid(
                         row=i, column=0, pady=6, sticky="w")

            var = tk.StringVar()
            self.vars[var_name] = var
            entry = tk.Entry(inp_frame, textvariable=var, font=FONT,
                             bg=ACCENT, fg=WHITE, insertbackground=WHITE,
                             relief="flat", width=14, bd=4)
            entry.grid(row=i, column=1, padx=10, pady=6)

            tk.Label(inp_frame, text=hint, bg=PANEL, fg="#888",
                     font=("Segoe UI", 9)).grid(row=i, column=2, sticky="w")

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=10)

        self._btn(btn_frame, "🔍  Predict", self._predict, GREEN,  0)
        self._btn(btn_frame, "🎲  Simulate",self._simulate, BLUE,  1)
        self._btn(btn_frame, "🗑  Clear",   self._clear,   ORANGE, 2)

        # ── Result section ────────────────────────────────────────────────────
        res_frame = tk.LabelFrame(self, text=" 📊  Prediction Result ",
                                  bg=PANEL, fg=WHITE, font=FONT_B,
                                  padx=20, pady=14, bd=2, relief="groove")
        res_frame.pack(fill="x", padx=20, pady=6)

        # Condition
        self.cond_var = tk.StringVar(value="—")
        self.cond_lbl = tk.Label(res_frame, textvariable=self.cond_var,
                                 bg=PANEL, fg=WHITE, font=("Segoe UI", 22, "bold"))
        self.cond_lbl.pack()

        # Anomaly
        self.anom_var = tk.StringVar(value="")
        tk.Label(res_frame, textvariable=self.anom_var,
                 bg=PANEL, fg=WHITE, font=FONT).pack(pady=(2, 10))

        # Confidence bars
        conf_title = tk.Label(res_frame, text="Confidence (Random Forest):",
                              bg=PANEL, fg="#aaa", font=("Segoe UI", 9))
        conf_title.pack(anchor="w")

        self.conf_bars = {}
        for cls, colour in [("Normal", GREEN), ("Overload", ORANGE), ("Fault", RED)]:
            row = tk.Frame(res_frame, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{cls:<10}", bg=PANEL, fg=WHITE,
                     font=("Courier", 10), width=10).pack(side="left")
            bar_bg = tk.Frame(row, bg="#333", width=280, height=16)
            bar_bg.pack(side="left", padx=4)
            bar_bg.pack_propagate(False)
            bar = tk.Frame(bar_bg, bg=colour, height=16, width=0)
            bar.pack(side="left", fill="y")
            pct_lbl = tk.Label(row, text="0.0%", bg=PANEL, fg=WHITE,
                               font=("Courier", 10))
            pct_lbl.pack(side="left")
            self.conf_bars[cls] = (bar, pct_lbl)

        # Footer
        tk.Label(self,
                 text="AIML Mini Project  |  scikit-learn RandomForest + IsolationForest",
                 bg=BG, fg="#555", font=("Segoe UI", 8)).pack(side="bottom", pady=6)

    def _btn(self, parent, text, cmd, colour, col):
        tk.Button(parent, text=text, command=cmd,
                  bg=colour, fg="black" if colour == ORANGE else "black",
                  font=FONT_B, relief="flat", padx=18, pady=8,
                  cursor="hand2",
                  activebackground=colour, activeforeground="black"
                  ).grid(row=0, column=col, padx=8)

    # ── Background training ───────────────────────────────────────────────────
    def _train_in_background(self):
        def _train():
            self.model.train(verbose=False)
            self.trained = True
            self.status_var.set("✅  Model ready — enter sensor values and click Predict")
            self.status_lbl.configure(bg=GREEN)
        threading.Thread(target=_train, daemon=True).start()

    # ── Actions ───────────────────────────────────────────────────────────────
    def _predict(self):
        if not self.trained:
            messagebox.showwarning("Please wait", "Model is still training …")
            return

        try:
            voltage     = float(self.vars["voltage_var"].get())
            load        = float(self.vars["load_var"].get())
            temperature = float(self.vars["temperature_var"].get())
            frequency   = float(self.vars["frequency_var"].get())
        except ValueError:
            messagebox.showerror("Input Error",
                                 "All fields must be numeric. Please check your inputs.")
            return

        try:
            validate_inputs(voltage, load, temperature, frequency)
        except ValueError as e:
            messagebox.showerror("Range Error", str(e))
            return

        result    = self.model.predict(voltage, load, temperature, frequency)
        condition = result["condition"]
        anomaly   = result["anomaly"]
        conf      = result["confidence"]

        colour_map = {"Normal": GREEN, "Overload": ORANGE, "Fault": RED}
        colour = colour_map.get(condition, WHITE)

        emoji = condition_emoji(condition)
        self.cond_var.set(f"{emoji}  {condition}")
        self.cond_lbl.configure(fg=colour)

        if anomaly:
            self.anom_var.set("🔴  ANOMALY DETECTED — abnormal reading flagged!")
        else:
            self.anom_var.set("🟢  No anomaly — reading within normal bounds.")

        # Update confidence bars
        for cls, (bar, lbl) in self.conf_bars.items():
            pct = conf.get(cls, 0)
            width = int(pct / 100 * 280)
            bar.configure(width=max(width, 1))
            lbl.configure(text=f"{pct:.1f}%")

        # Update status bar colour
        self.status_var.set(
            f"  Last prediction: {condition}  |  "
            f"Anomaly: {'YES' if anomaly else 'NO'}"
        )
        self.status_lbl.configure(bg=colour)

    def _simulate(self):
        s = simulate_sensor("random")
        self.vars["voltage_var"].set(s["voltage"])
        self.vars["load_var"].set(s["load"])
        self.vars["temperature_var"].set(s["temperature"])
        self.vars["frequency_var"].set(s["frequency"])
        if self.trained:
            self._predict()

    def _clear(self):
        for var in self.vars.values():
            var.set("")
        self.cond_var.set("—")
        self.anom_var.set("")
        for cls, (bar, lbl) in self.conf_bars.items():
            bar.configure(width=0)
            lbl.configure(text="0.0%")
        self.status_var.set("✅  Model ready — enter sensor values and click Predict")
        self.status_lbl.configure(bg=GREEN)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = TransformerApp()
    app.mainloop()
