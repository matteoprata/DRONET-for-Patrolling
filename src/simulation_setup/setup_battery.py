

from src.constants import IndependentVariable as indv
from src.constants import OnlinePatrollingProtocol as pol
from src.constants import PrecomputedPatrollingProtocol as pol2
from src.constants import ToleranceScenario, PositionScenario

# SEED x ALGO (N_DRONES)

comp_dims = {indv.SEED: range(15, 30),  # [0, 2, 3, 4, 5],
             indv.DRONE_PATROLLING_POLICY: [pol2.INFOCOM,
                                            pol2.BARTOLINI,
                                            ]
             }

indv_vary = {
    indv.DRONES_NUMBER: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # [1, 2, 5, 10, 15, 20, 25],
}

indv_fixed = {
    indv.DRONE_SPEED: 5,
    indv.DRONES_NUMBER: 10,
    indv.TARGETS_NUMBER: 40,
    indv.TARGETS_TOLERANCE_FIXED: 2,
    indv.TARGETS_TOLERANCE_SCENARIO: ToleranceScenario.GAUSSIAN,
    indv.TARGETS_POSITION_SCENARIO: PositionScenario.UNIFORM,
}
