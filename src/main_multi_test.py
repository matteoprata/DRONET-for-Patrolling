
import argparse
import traceback

import src.constants as cst
import src.utilities.utilities as util
from src.world_entities.simulator_patrolling import PatrollingSimulator
from src.config import Configuration
from src.constants import IndependentVariable as indv

import copy


def main_multi_test(configuration: Configuration):
    """ Experiment (trained) policies. """

    # 1. Declare independent variables and their domain
    # 2. Declare what independent variable varies at this execution and what stays fixed

    stp = cst.Setups[configuration.SETUP_NAME].value

    patrolling_args = []
    indv_fixed_original = {k: stp.indv_fixed[k] for k in stp.indv_fixed}
    for a in stp.comp_dims[indv.DRONE_PATROLLING_POLICY]:
        for s in stp.comp_dims[indv.SEED]:
            for x_var_k in stp.indv_vary:  # iterates over the independent variable
                X_var = stp.indv_vary[x_var_k]
                for x in X_var:  # iterates over the values of the i-th independent variable
                    stp.indv_fixed[x_var_k] = x
                    stp.indv_fixed[indv.DRONE_PATROLLING_POLICY] = a
                    stp.indv_fixed[indv.SEED] = s

                    # stp.indv_fixed is now fully formed

                    configuration = copy.deepcopy(configuration)
                    for var, val in stp.indv_fixed.items():
                        # NOTICE: assumes that the keys in the setup indv_fixed dict have the same name as the values in cf
                        setattr(configuration, var.name, val)

                    # configuration values
                    patrolling_args.append([configuration])
                    stp.indv_fixed = {k: indv_fixed_original[k] for k in indv_fixed_original}  # reset the change

    if configuration.IS_PARALLEL_EXECUTION:
        print("Parallel on {} cores".format(cst.N_CORES))
        util.execute_parallel(main_safe_execution, patrolling_args, cst.N_CORES)
    else:
        print("Sequential.")
        for c in patrolling_args:
            main_safe_execution(c[0])


def main_safe_execution(configuration):
    try:
        configuration.run_parameters_sanity_check()
        print("\nExecuting > {}\n".format(configuration.conf_description()))
        sim = PatrollingSimulator(configuration)
        sim.run_testing_loop()

    except:
        print("Could not solve problem for {}!".format(configuration.conf_description()))
        trace = traceback.format_exc()
        print("/n", trace)


def parser_cl_arguments(configuration: Configuration):
    """ Parses the arguments for the command line. """

    parser = argparse.ArgumentParser(description='Patrolling Simulator arguments:')

    parser.add_argument('-set', '--SETUP_NAME', type=str)
    parser.add_argument('-par', '--IS_PARALLEL_EXECUTION', default=0, type=int)
    parser.add_argument('-pl',  '--PLOT_SIM', default=configuration.PLOT_SIM, type=int)

    # parsing arguments from cli
    args = vars(parser.parse_args())

    print("Setting parameters...")
    configuration.SETUP_NAME = args["SETUP_NAME"].upper()
    configuration.IS_PARALLEL_EXECUTION = bool(args["IS_PARALLEL_EXECUTION"])
    configuration.PLOT_SIM = bool(args["PLOT_SIM"])


# python -m src.main_multi_test -set IOT -par 0 -pl 1

if __name__ == "__main__":
    # conf = Configuration()
    # parser_cl_arguments(conf)
    # main_multi_test(conf)

    configuration = Configuration()

    configuration.EPISODE_DURATION = 10000

    configuration.SETUP_NAME = 'SETUP0'
    configuration.IS_PARALLEL_EXECUTION = True

    # configuration.SETUP_NAME = 'SETUP_SOLO'
    # configuration.IS_PARALLEL_EXECUTION = False

    configuration.PLOT_SIM = not configuration.IS_PARALLEL_EXECUTION
    main_multi_test(configuration)