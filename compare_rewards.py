import numpy as np

"""
compare_rewards.py

Simple training-curve comparison between baseline and PressLight.

- Loads:
    * baseline_rewards.npy
    * presslight_rewards.npy
- Prints:
    * Mean training reward for each method.
    * Last few episode rewards (to see end-of-training behavior).

This works on a single training run and does not do proper statistics;
it's just a quick sanity check on training curves.
"""


baseline = np.load("baseline_rewards.npy")
press = np.load("presslight_rewards.npy")

print("Baseline - mean reward:", baseline.mean())
print("PressLight - mean reward:", press.mean())
print("Baseline - last 3 episodes:", baseline[-3:])
print("PressLight - last 3 episodes:", press[-3:])
