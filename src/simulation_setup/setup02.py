
from src.utilities.constants import IndependentVariable as indv
from src.utilities.constants import Mobility as pol


comp_dims = {indv.SEED: [0],
             indv.ALGORITHM: [pol.RANDOM_MOVEMENT]}

indv_vary = {
    indv.DRONES_SPEED: [5],
    indv.DRONES_NUMBER: [4],
    indv.TARGETS_NUMBER: [10],
    indv.TARGETS_TOLERANCE: [-0.5, 0, 0.2],
}

indv_fixed = {
    indv.DRONES_SPEED: 15,
    indv.DRONES_NUMBER: 4,
    indv.TARGETS_NUMBER: 10,
    indv.TARGETS_TOLERANCE: 0,
}

