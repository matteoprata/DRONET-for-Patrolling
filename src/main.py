
from src.simulation.simulator_patrolling import PatrollingSimulator
import src.utilities.config as config


def main():
    """ the place where to run simulations and experiments. """

    sim = PatrollingSimulator(sim_peculiarity="", drone_mobility=config.Mobility.DECIDED)
    sim.run()


if __name__ == "__main__":
    main()
