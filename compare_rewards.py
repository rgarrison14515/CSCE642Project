import numpy as np

baseline = np.load("baseline_rewards.npy")
press = np.load("presslight_rewards.npy")

print("Baseline - mean reward:", baseline.mean())
print("PressLight - mean reward:", press.mean())
print("Baseline - last 3 episodes:", baseline[-3:])
print("PressLight - last 3 episodes:", press[-3:])
