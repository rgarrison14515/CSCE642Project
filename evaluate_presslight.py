import torch
import numpy as np
from agents.dqn_presslight import DQN
from sumo_env import SumoEnv

def evaluate(model_path="presslight_model.pt", episodes=5):
    env = SumoEnv(
        cfg_path="sumo_env/configs/3x3.sumocfg",
        tls_id="B1",
        use_gui=False,
        mode="presslight",
    )

    # Load model
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.num_phases

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                a = int(model(x).argmax().item())

            state, reward, done, _ = env.step(a)
            total_reward += reward

        results.append(total_reward)
        print(f"[Eval] Episode {ep+1}/{episodes} Reward={total_reward:.1f}")


    env.close()
    return results


if __name__ == "__main__":
    evaluate()
