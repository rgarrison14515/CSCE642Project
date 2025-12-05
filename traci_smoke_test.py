import traci
import sumolib

CFG_PATH = "sumo_env/configs/3x3.sumocfg"

"""
traci_smoke_test.py

Minimal TraCI connectivity check.

- Starts SUMO (or sumo-gui) with the given .sumocfg.
- Steps the simulation a small number of times.
- Prints step indices and then cleanly closes the TraCI connection.

Used mainly during initial setup to debug connection issues.
"""



def main():
    sumo_binary = sumolib.checkBinary("sumo-gui")  # or "sumo" if you prefer no GUI

    print("Starting SUMO with:", sumo_binary, "-c", CFG_PATH)
    traci.start([sumo_binary, "-c", CFG_PATH, "--step-length", "1.0"])

    for i in range(10):
        print("step", i)
        traci.simulationStep()

    print("Closing TraCI.")
    traci.close()

if __name__ == "__main__":
    main()
