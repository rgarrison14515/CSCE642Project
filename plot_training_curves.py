# plot_training_curves.py
import numpy as np
import matplotlib.pyplot as plt

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
