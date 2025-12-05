# plot_training_curves.py
import numpy as np
import matplotlib.pyplot as plt

"""
plot_training_curves.py

Generates plots for training behavior of baseline and PressLight.

- Loads rewards and/or losses from:
    * baseline_rewards.npy / baseline_losses.npy
    * presslight_rewards.npy / presslight_losses.npy
- Plots:
    * Reward vs. episode (optionally smoothed).
    * Loss vs. training step (optional).

Use this to create figures for the report.
"""


def main():
    baseline_rewards = np.load("baseline_rewards.npy")
    press_rewards = np.load("presslight_rewards.npy")

    episodes_baseline = np.arange(1, len(baseline_rewards) + 1)
    episodes_press = np.arange(1, len(press_rewards) + 1)

    plt.figure()
    plt.plot(episodes_baseline, baseline_rewards, label="Baseline (queue reward)")
    plt.plot(episodes_press, press_rewards, label="PressLight (pressure reward)")
    plt.xlabel("Episode")
    plt.ylabel("Total episode reward")
    plt.title("Training Reward Curves (Baseline vs PressLight)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_curves.png")
    plt.show()

if __name__ == "__main__":
    main()
