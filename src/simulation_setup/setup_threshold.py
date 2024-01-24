

from src.constants import IndependentVariable as indv
from src.constants import OnlinePatrollingProtocol as pol
from src.constants import PrecomputedPatrollingProtocol as pol2
from src.constants import ToleranceScenario, PositionScenario

# SEED x ALGO (N_DRONES)

comp_dims = {indv.SEED: range(0, 15),  # [0, 2, 3, 4, 5],
             indv.DRONE_PATROLLING_POLICY: [pol2.INFOCOM,
                                            pol2.PARTITION,
                                            pol2.CYCLE,
                                            pol.GO_MAX_AOI,
                                            pol.GO_MIN_RESIDUAL,
                                            ]
             }

indv_vary = {
    indv.TARGETS_TOLERANCE_FIXED: [1, 1.2, 1.4, 1.6, 1.8, 2]
}

indv_fixed = {
    indv.DRONE_SPEED: 5,
    indv.DRONES_NUMBER: 10,
    indv.TARGETS_NUMBER: 40,
    indv.TARGETS_TOLERANCE_FIXED: 1,
    indv.TARGETS_TOLERANCE_SCENARIO: ToleranceScenario.GAUSSIAN,
    indv.TARGETS_POSITION_SCENARIO: PositionScenario.CLUSTERED,
}
