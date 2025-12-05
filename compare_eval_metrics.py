# compare_eval_metrics.py
import numpy as np

"""
compare_eval_metrics.py

Compares evaluation-time queue and delay metrics between methods.

- Loads:
    * baseline_eval_queues.npy, baseline_eval_delays.npy
    * presslight_eval_queues.npy, presslight_eval_delays.npy
- Prints:
    * Average queue length and delay for each method.
    * Per-episode queue and delay arrays.

Used for the final quantitative comparison in the report (single-intersection case).
"""


baseline_q = np.load("baseline_eval_queues.npy")
baseline_d = np.load("baseline_eval_delays.npy")

press_q = np.load("presslight_eval_queues.npy")
press_d = np.load("presslight_eval_delays.npy")

print("=== Baseline ===")
print("  Avg queue:", baseline_q.mean())
print("  Avg delay:", baseline_d.mean())
print("  Per-episode queues:", baseline_q)
print("  Per-episode delays:", baseline_d)

print("\n=== PressLight ===")
print("  Avg queue:", press_q.mean())
print("  Avg delay:", press_d.mean())
print("  Per-episode queues:", press_q)
print("  Per-episode delays:", press_d)
