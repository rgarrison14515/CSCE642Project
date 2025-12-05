# CSCE 642 – SUMO Traffic Signal RL Env

This repo contains a SUMO + TraCI environment for learning traffic light control on a small 3×3 grid network. We use it to implement and compare:

- A **baseline DQN controller** with a simple queue-based reward.
- A **PressLight-style DQN controller** with max-pressure state and reward.

The long-term goal is to also explore MPLight, but this codebase currently focuses on the single-intersection PressLight baseline vs. control comparison.

---

## Project Structure

```text
CSCE642Project/
├── sumo_env/
│   ├── __init__.py           # Exposes SumoEnv
│   ├── env.py                # SumoEnv (single TLS controller, baseline/PressLight modes)
│   ├── configs/
│   │   └── 3x3.sumocfg       # SUMO configuration (3×3 grid, TLS at junction B1)
│   ├── net/
│   │   └── 3x3.net.xml       # 3×3 grid network (SUMO net)
│   └── routes/
│       └── 3x3.rou.xml       # Route / flow definitions (medium demand)
│
├── agents/
│   ├── dqn_baseline.py       # Baseline DQN (halted-vehicle reward, mode="baseline")
│   └── dqn_presslight.py     # PressLight-style DQN (pressure reward, mode="presslight")
│
├── test_env_random.py        # Random policy sanity check on SumoEnv
├── presslight_smoke_test.py  # Smoke test for PressLight state/reward construction
├── traci_smoke_test.py       # Minimal TraCI connectivity check
│
├── evaluate_baseline.py      # Loads baseline model, runs eval, logs queue/delay
├── evaluate_presslight.py    # Same for PressLight model
├── compare_rewards.py        # Prints training reward summary for both methods
├── compare_eval_metrics.py   # Compares eval-time queue/delay metrics
├── plot_training_curves.py   # (Optional) plots reward/loss over episodes
├── summarize_checkpoints.py  # Summarizes reward at episode cutoffs
│
├── requirements.txt
├── .gitignore
└── README.md
