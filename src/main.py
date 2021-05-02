
from src.simulation.simulator_patrolling import PatrollingSimulator
import src.utilities.config as config


def main():
    """ the place where to run simulations and experiments. """

    sim = PatrollingSimulator(sim_description="new2")
    sim.run()


if __name__ == "__main__":
    main()
