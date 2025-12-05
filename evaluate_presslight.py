# evaluate_presslight.py
import numpy as np
import torch
import traci

from agents.dqn_presslight import DQN
from sumo_env import SumoEnv

"""
evaluate_presslight.py

Evaluation script for the trained PressLight-style DQN model.

- Loads presslight_model.pt.
- Runs greedy evaluation episodes using SumoEnv(mode="presslight").
- Computes and prints:
    * Total reward per episode (max-pressure-based).
    * Average queue length and delay (same metric as baseline, for comparison).
- Saves:
    * presslight_eval_queues.npy
    * presslight_eval_delays.npy
"""



def evaluate_presslight(
    model_path: str = "presslight_model.pt",
    episodes: int = 5,
):
    env = SumoEnv(
        cfg_path="sumo_env/configs/3x3.sumocfg",
        tls_id="B1",
        use_gui=False,
        mode="presslight",
    )

    # Build model with correct dimensions
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.num_phases

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    episode_rewards = []
    episode_avg_queues = []
    episode_avg_delays = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        total_queue = 0.0
        total_delay = 0.0
        steps = 0

        while not done:
            # greedy action
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = model(x)
                action = int(q_vals.argmax(dim=1).item())

            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            # ---- metrics: queue + delay per timestep ----
            step_queue = 0.0
            step_delay = 0.0

            for lane in env.lanes:
                halted = traci.lane.getLastStepHaltingNumber(lane)
                step_queue += halted

                v = traci.lane.getLastStepMeanSpeed(lane)
                vmax = traci.lane.getMaxSpeed(lane)
                nveh = traci.lane.getLastStepVehicleNumber(lane)

                if vmax > 0 and nveh > 0 and v >= 0:
                    step_delay += (1.0 - v / vmax) * nveh

            total_queue += step_queue
            total_delay += step_delay

        avg_queue = total_queue / max(1, steps)
        avg_delay = total_delay / max(1, steps)

        episode_rewards.append(total_reward)
        episode_avg_queues.append(avg_queue)
        episode_avg_delays.append(avg_delay)

        print(
            f"[PressLight Eval] Episode {ep+1}/{episodes} "
            f"Reward={total_reward:.1f}, AvgQueue={avg_queue:.3f}, AvgDelay={avg_delay:.3f}"
        )

    env.close()

    np.save("presslight_eval_rewards.npy", np.array(episode_rewards))
    np.save("presslight_eval_queues.npy", np.array(episode_avg_queues))
    np.save("presslight_eval_delays.npy", np.array(episode_avg_delays))

    return episode_rewards, episode_avg_queues, episode_avg_delays


if __name__ == "__main__":
    evaluate_presslight()
