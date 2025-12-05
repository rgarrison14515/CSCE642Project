import numpy as np

baseline_q = np.load("baseline_eval_queues.npy")
press_q = np.load("presslight_eval_queues.npy")

print("Baseline  avg queue:", baseline_q.mean())
print("PressLight avg queue:", press_q.mean())
print("Baseline  per-episode queues:", baseline_q)
print("PressLight per-episode queues:", press_q)
