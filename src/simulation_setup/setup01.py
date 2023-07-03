
from src.constants import IndependentVariable as indv
from src.constants import OnlinePatrollingProtocol as pol

comp_dims = {indv.SEED: range(5),
             indv.DRONE_PATROLLING_POLICY: [pol.RANDOM_MOVEMENT,
                                            pol.GO_MAX_AOI,
                                            pol.GO_MIN_RESIDUAL,
                                            pol.GO_MIN_SUM_RESIDUAL,
                                            ]
             }

indv_vary = {
    indv.DRONE_SPEED: [5, 10, 15, 20, 25],
    indv.DRONES_NUMBER: [1, 2, 3, 4, 5, 6],
    indv.TARGETS_NUMBER: [4, 6, 8, 10, 12, 14],
    indv.TARGETS_TOLERANCE_FIXED: [-.4, -.3, -.2, 0, .2, .3, .4],
}

indv_fixed = {
    indv.DRONE_SPEED: 15,
    indv.DRONES_NUMBER: 4,
    indv.TARGETS_NUMBER: 10,
    indv.TARGETS_TOLERANCE_FIXED: 0,
}
