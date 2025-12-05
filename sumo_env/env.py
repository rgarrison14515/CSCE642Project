import os
import numpy as np
import traci
import sumolib


class SumoEnv:
    def __init__(
        self,
        cfg_path: str,
        tls_id: str,
        step_length: float = 1.0,
        action_interval: int = 5,
        warmup_steps: int = 100,
        max_steps: int = 3600,
        use_gui: bool = True,
        mode: str = "baseline",  # "baseline" or "presslight"
    ):
        

        """
        PressLight mode implements the state and reward from Wei et al. (KDD 2019). 
        State = [incoming lane segments, outgoing lane queues, phase], reward = 
        -intersection pressure, where movement pressure is w(l,m) = x(l) - x(m).
        """

        """
        SUMO environment that controls a single traffic light.

        cfg_path: path to .sumocfg (e.g. sumo_env/configs/3x3.sumocfg)
        tls_id:   traffic light ID to control (e.g. 'B1')
        mode:     "baseline"  -> simple queue-based state/reward
                  "presslight" -> pressure-based state/reward
        """
        self.cfg_path = cfg_path
        self.tls_id = tls_id
        self.step_length = step_length
        self.action_interval = action_interval
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.use_gui = use_gui
        self.mode = mode

        self.sumo_binary = sumolib.checkBinary("sumo-gui" if use_gui else "sumo")
        self.lanes = None
        self.num_phases = None
        self.sim_step = 0

        # For PressLight: list of (incoming_lane, outgoing_lane) controlled by this TLS
        self.movements = []

    # ---------- startup / shutdown ----------

    def _start_traci(self):
        if traci.isLoaded():
            traci.close()

        args = [
            self.sumo_binary,
            "-c", self.cfg_path,
            "--step-length", str(self.step_length),
            "--no-step-log", "true",
            "--error-log", "sumo_error.log",  # log SUMO errors here
        ]
        print("Starting SUMO with args:", args)

        traci.start(args)
        self.sim_step = 0

    def close(self):
        if traci.isLoaded():
            traci.close()

    # ---------- RL-style API ----------

    def reset(self):
        """(Re)start SUMO, warm up traffic, and return initial state vector."""
        self._start_traci()

        print(f"Warmup: up to {self.warmup_steps} steps")
        for i in range(self.warmup_steps):
            # If SUMO has no vehicles now and none expected, don't keep stepping
            if traci.simulation.getMinExpectedNumber() == 0:
                print(f"  stopping warmup early at step {i}: no vehicles expected")
                break

            traci.simulationStep()
            self.sim_step += 1

        print("After warmup:")
        print("  sim_step:", self.sim_step)
        print("  minExpectedNumber:", traci.simulation.getMinExpectedNumber())
        print("  vehicles in sim:", traci.vehicle.getIDList())

        # cache lanes controlled by this TL and number of phases
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self.lanes = list(dict.fromkeys(lanes))

        # Get the full program logic and infer number of phases
        programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
        if not programs:
            raise RuntimeError(f"No signal program found for TLS {self.tls_id}")

        logic = programs[0]  # use the first (default) program
        self.num_phases = len(logic.phases)

        # For PressLight: build list of lane-to-lane movements (l, m)
        # getControlledLinks returns, for each link index, a list of connections:
        #   (incoming_lane, outgoing_lane, via_lane)
        self.movements = []
        links = traci.trafficlight.getControlledLinks(self.tls_id)
        for conn_list in links:
            if not conn_list:
                continue
            inc_lane, out_lane, _via = conn_list[0]
            self.movements.append((inc_lane, out_lane))

        # dedupe (l, m) while preserving order
        self.movements = list(dict.fromkeys(self.movements))

        print("  TLS", self.tls_id, "controls lanes:", self.lanes)
        print("  num_phases:", self.num_phases)
        print("  mode:", self.mode)
        if self.mode == "presslight":
            print("  movements (incoming -> outgoing):")
            for (l, m) in self.movements:
                print("    ", l, "->", m)

        return self._get_state()

    def step(self, action: int):
        """
        Apply an action (phase index), advance the sim for action_interval steps,
        and return (state, reward, done, info).
        """
        if self.num_phases is None:
            raise RuntimeError("Call reset() before step().")

        phase = int(action) % self.num_phases
        traci.trafficlight.setPhase(self.tls_id, phase)

        total_reward = 0.0

        for _ in range(self.action_interval):
            traci.simulationStep()
            self.sim_step += 1

            total_reward += self._get_reward()

            if traci.simulation.getMinExpectedNumber() == 0:
                break

        state = self._get_state()
        done = (
            self.sim_step >= self.max_steps
            or traci.simulation.getMinExpectedNumber() == 0
        )
        info = {
            "sim_step": self.sim_step,
            "current_phase": traci.trafficlight.getPhase(self.tls_id),
        }

        return state, total_reward, done, info

    # ---------- state / reward ----------

    def _get_state(self):
        """
        State depends on mode:

        - baseline:
            [ vehicle_count(lane_0), ..., vehicle_count(lane_n), current_phase ]

        - presslight:
            For each movement (l, m):
                x(l), x(m)
            ... then current_phase appended.
        """
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        if self.mode == "presslight" and self.movements:
            state_vals = []
            for (inc, out) in self.movements:
                x_in = traci.lane.getLastStepVehicleNumber(inc)
                x_out = traci.lane.getLastStepVehicleNumber(out)
                state_vals.extend([x_in, x_out])

            state_vals.append(current_phase)
            return np.array(state_vals, dtype=np.float32)

        # fallback / baseline
        lane_counts = [
            traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in self.lanes
        ]
        lane_counts.append(current_phase)
        return np.array(lane_counts, dtype=np.float32)

    def _get_reward(self):
        """
        Reward depends on mode:

        - baseline:
            negative total number of stopped vehicles on all lanes
        - presslight:
            negative total pressure:
                sum over movements (l, m) of (x(l) - x(m))
        """
        if self.mode == "presslight" and self.movements:
            total_pressure = 0.0
            for (inc, out) in self.movements:
                x_in = traci.lane.getLastStepVehicleNumber(inc)
                x_out = traci.lane.getLastStepVehicleNumber(out)
                total_pressure += (x_in - x_out)
            return -float(total_pressure)

        # baseline
        halted = [
            traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self.lanes
        ]
        return -float(sum(halted))
    
    def get_total_queue(self) -> float:
        """
        Returns the total number of halted vehicles on all lanes
        controlled by this traffic light. Useful as a shared
        evaluation metric for different agents.
        """
        if self.lanes is None:
            return 0.0
        halted = [
            traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self.lanes
        ]
        return float(sum(halted))
