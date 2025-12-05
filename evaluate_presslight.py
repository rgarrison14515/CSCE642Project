import numpy as np
import torch

from agents.dqn_presslight import DQN
from sumo_env import SumoEnv


def evaluate_presslight(
    model_path: str = "presslight_model.pt",
    cfg_path: str = "sumo_env/configs/3x3.sumocfg",
    tls_id: str = "B1",
    episodes: int = 5,
    max_steps_per_episode: int = 500,
):
    env = SumoEnv(
        cfg_path=cfg_path,
        tls_id=tls_id,
        use_gui=False,
        mode="presslight",
    )

    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.num_phases

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    episode_rewards = []
    episode_avg_queues = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        queue_sum = 0.0
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(model(x).argmax().item())

            state, reward, done, _ = env.step(action)
            total_reward += reward

            queue_sum += env.get_total_queue()
            step_count += 1

        avg_queue = queue_sum / max(1, step_count)
        episode_rewards.append(total_reward)
        episode_avg_queues.append(avg_queue)

        print(
            f"[PressLight Eval] Episode {ep+1}/{episodes} "
            f"Reward={total_reward:.1f}, AvgQueue={avg_queue:.2f}"
        )

    env.close()

    np.save("presslight_eval_rewards.npy", np.array(episode_rewards))
    np.save("presslight_eval_queues.npy", np.array(episode_avg_queues))

    return episode_rewards, episode_avg_queues


if __name__ == "__main__":
    evaluate_presslight()
