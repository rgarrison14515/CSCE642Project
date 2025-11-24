import numpy as np
from sumo_env.env import SumoEnv   # adjust if env.py is in a subfolder

CFG_PATH = "sumo_env/configs/3x3.sumocfg"
TLS_ID = "B1"   # from your list_tls_ids output

def main():
    env = SumoEnv(
        cfg_path=CFG_PATH,
        tls_id=TLS_ID,
        step_length=1.0,
        action_interval=5,
        warmup_steps=100,
        max_steps=500,
        use_gui=True
    )

    state = env.reset()
    print("Initial state:", state)

    done = False
    episode_reward = 0.0

    while not done:
        action = np.random.randint(0, env.num_phases)
        state, reward, done, info = env.step(action)
        episode_reward += reward

        print(f"Step {info['sim_step']:4d} | action={action} "
              f"| reward={reward:6.2f} | phase={info['current_phase']}")

    print("Episode finished. Total reward:", episode_reward)
    env.close()

if __name__ == "__main__":
    main()
