

from src.constants import IndependentVariable as indv
from src.constants import OnlinePatrollingProtocol as pol
from src.constants import PrecomputedPatrollingProtocol as pol2
from src.constants import ToleranceScenario, PositionScenario

# SEED x ALGO (N_DRONES)

comp_dims = {indv.SEED: [0, 2, 3],
             indv.DRONE_PATROLLING_POLICY: [
                                            # pol.RANDOM_MOVEMENT,
                                            # pol2.MULTI_TSP,
                                            # pol.GO_MAX_AOI,
                                            # pol.GO_MIN_SUM_RESIDUAL,
                                            # pol.GO_MIN_RESIDUAL,
                                            # pol.CLUSTER_GO_MIN_RESIDUAL,
                                            # pol2.PEPPE_CLUSTERING,
                                            pol2.OURS
                                            ]
             }

indv_vary = {
    # indv.DRONE_SPEED: [5, 10, 15, 20, 25],
    indv.DRONES_NUMBER: [15, 20],
}

indv_fixed = {
    indv.DRONE_SPEED: 10,
    indv.DRONES_NUMBER: 4,
    indv.TARGETS_NUMBER: 90,
    indv.TARGETS_TOLERANCE_FIXED: 100,
    indv.TARGETS_TOLERANCE_SCENARIO: ToleranceScenario.CONSTANT,
    indv.TARGETS_POSITION_SCENARIO: PositionScenario.CLUSTERED,
}
