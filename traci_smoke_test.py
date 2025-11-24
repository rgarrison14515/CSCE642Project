import traci
import sumolib

CFG_PATH = "sumo_env/configs/3x3.sumocfg"

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
