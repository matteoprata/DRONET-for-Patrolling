
from src.constants import IndependentVariable as indv
from src.constants import PatrollingProtocol as pol


comp_dims = {indv.SEED: [5],
             indv.DRONE_PATROLLING_POLICY: [pol.RANDOM_MOVEMENT,
                                            pol.GO_MIN_SUM_RESIDUAL
                                            ]}

indv_vary = {
    indv.DRONE_SPEED: [5, 10, 15],
    # indv.DRONES_NUMBER: [2, 3, 4, 5, 6],
    # indv.TARGETS_NUMBER: [10],
    # indv.TARGETS_TOLERANCE_SCALE: [-0.5, 0, 0.2],
}

indv_fixed = {
    indv.DRONE_SPEED: 15,
    indv.DRONES_NUMBER: 1,
    indv.TARGETS_NUMBER: 3,
    indv.TARGETS_TOLERANCE_FIXED: 0,
}

