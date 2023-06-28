

from src.constants import IndependentVariable as indv
from src.constants import PatrollingProtocol as pol
from src.constants import PrecomputedPatrollingProtocol as pol2

# SEED x ALGO (N_DRONES)

comp_dims = {indv.SEED: range(2),
             indv.DRONE_PATROLLING_POLICY: [pol.RANDOM_MOVEMENT,
                                            pol2.MULTI_TSP,
                                            pol.GO_MAX_AOI,
                                            pol.GO_MIN_SUM_RESIDUAL,
                                            pol.GO_MIN_RESIDUAL
                                            ]
             }

indv_vary = {
    # indv.DRONE_SPEED: [5, 10, 15, 20, 25],
    indv.DRONES_NUMBER: [1, 2, 3, 4, 5, 6],
}

indv_fixed = {
    indv.DRONE_SPEED: 15,
    indv.DRONES_NUMBER: 4,
    indv.TARGETS_NUMBER: 30,
    indv.TARGETS_TOLERANCE_FIXED: 300,
}

