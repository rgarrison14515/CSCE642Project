# presslight_smoke_test.py
from sumo_env import SumoEnv

def main():
    env = SumoEnv(
        cfg_path="sumo_env/configs/3x3.sumocfg",
        tls_id="B1",
        use_gui=True,          # turn on GUI so you can see it
        mode="presslight",     # <--- new mode
        warmup_steps=50,
        action_interval=5,
    )

    state = env.reset()
    print("Initial PressLight state shape:", state.shape)
    print("Initial PressLight state:", state)

    # Take a few random actions just to see nothing crashes
    import random
    total_reward = 0.0

    for t in range(10):
        action = random.randrange(env.num_phases)
        s, r, done, info = env.step(action)
        total_reward += r
        print(f"t={t}, action={action}, reward={r:.2f}, state_shape={s.shape}")

        if done:
            print("Episode ended early at t=", t)
            break

    print("Total reward over 10 steps:", total_reward)
    env.close()


if __name__ == "__main__":
    main()
