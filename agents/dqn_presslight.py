# agents/dqn_presslight.py
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sumo_env import SumoEnv


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)


def train_presslight(
    cfg_path="sumo_env/configs/3x3.sumocfg",
    tls_id="B1",
    episodes=10,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_capacity=10000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    target_update_every=5,
    max_steps_per_episode=500,
):
    # Only difference vs baseline is mode="presslight"
    env = SumoEnv(
        cfg_path=cfg_path,
        tls_id=tls_id,
        use_gui=False,
        mode="presslight",
    )

    # --- logging containers ---
    episode_rewards = []
    losses = []
    epsilons = []

    # init env once to infer dims
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = env.num_phases

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start

    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0.0

        for t in range(max_steps_per_episode):
            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(
                        state, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    q_vals = policy_net(s_t)
                    action = int(q_vals.argmax(dim=1).item())

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            replay.push(state, action, reward, next_state, done)
            state = next_state

            # learn
            if len(replay) >= batch_size:
                batch_s, batch_a, batch_r, batch_ns, batch_d = replay.sample(batch_size)

                batch_s = torch.tensor(batch_s, dtype=torch.float32, device=device)
                batch_a = torch.tensor(batch_a, dtype=torch.int64, device=device)
                batch_r = torch.tensor(batch_r, dtype=torch.float32, device=device)
                batch_ns = torch.tensor(batch_ns, dtype=torch.float32, device=device)
                batch_d = torch.tensor(batch_d, dtype=torch.float32, device=device)

                q_values = policy_net(batch_s).gather(
                    1, batch_a.unsqueeze(1)
                ).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_net(batch_ns).max(dim=1)[0]
                    target = batch_r + gamma * (1.0 - batch_d) * next_q_values

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # --- log loss ---
                losses.append(loss.item())

            if done:
                break

        # decay epsilon AFTER each episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # target network update
        if (ep + 1) % target_update_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # --- log episode-level stuff ---
        episode_rewards.append(episode_reward)
        epsilons.append(epsilon)

        print(
            f"[PressLight] Episode {ep+1}/{episodes} | reward={episode_reward:.1f} | epsilon={epsilon:.3f}"
        )

    # --- save logs & model at end ---
    np.save("presslight_rewards.npy", np.array(episode_rewards, dtype=np.float32))
    np.save("presslight_losses.npy", np.array(losses, dtype=np.float32))
    np.save("presslight_epsilons.npy", np.array(epsilons, dtype=np.float32))

    torch.save(policy_net.state_dict(), "presslight_model.pt")

    env.close()
    return episode_rewards


if __name__ == "__main__":
    train_presslight()
