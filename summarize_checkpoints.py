import numpy as np

"""
summarize_checkpoints.py

Prints coarse summaries of training progress at different episode cutoffs.

- For baseline and PressLight, it computes:
    * Mean reward up to episodes 10, 25, 50, 100, ...
    * Mean reward over the last few episodes before each cutoff.

Helpful for checking convergence trends without looking at full plots.
"""


baseline = np.load("baseline_rewards.npy")
press = np.load("presslight_rewards.npy")

checkpoints = [10, 25, 50, 100]

print("=== Baseline ===")
for k in checkpoints:
    # mean reward up to episode k
    mean_k = baseline[:k].mean()
    # or mean of the last 5 episodes up to k
    tail = baseline[max(0, k-5):k].mean()
    print(f"up to ep {k:3d}: mean={mean_k:8.1f}, last5={tail:8.1f}")

print("\n=== PressLight ===")
for k in checkpoints:
    mean_k = press[:k].mean()
    tail = press[max(0, k-5):k].mean()
    print(f"up to ep {k:3d}: mean={mean_k:8.1f}, last5={tail:8.1f}")
