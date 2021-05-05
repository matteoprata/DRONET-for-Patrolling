
from src.simulation.simulator_patrolling import PatrollingSimulator
import src.utilities.config as config
import sys

def main():
    """ the place where to run simulations and experiments. """
    description = sys.argv[1]
    sim = PatrollingSimulator(sim_description=description)
    sim.run()


if __name__ == "__main__":
    main()
