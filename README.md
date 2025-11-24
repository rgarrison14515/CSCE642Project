# CSCE 642 – SUMO Traffic Signal RL Env

This repo contains a SUMO + TraCI environment for a single traffic light on a 3x3 grid network. It will be used as the base for a baseline RL agent and later PressLight / MPLight implementations.

## Project Structure

```text
project/
├── sumo_env/
│   ├── __init__.py
│   ├── env.py                # SumoEnv class (single TLS controller)
│   ├── configs/
│   │   └── 3x3.sumocfg       # SUMO configuration file
│   ├── net/
│   │   └── 3x3.net.xml       # 3x3 grid network, TLS at junction B1
│   └── routes/
│       └── 3x3.rou.xml       # Route / flow definitions
├── test_env_random.py        # Runs a random policy on TLS B1
└── traci_smoke_test.py       # Simple TraCI sanity check
