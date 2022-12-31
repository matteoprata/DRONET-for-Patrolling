
from src.utilities.utilities import min_max_normalizer, euclidean_distance
from enum import Enum

from src.world_entities.drone import Drone
from src.world_entities.target import Target


class State:
    def __init__(self, features: list):
        self.features = features
        self.vector_value = None

        self.features_dict = dict()
        for f in features:
            self.features_dict[f.name] = f

    def vector(self, is_normalized=True):
        if self.vector_value is None:
            vec = []
            for f in self.features:
                vec += f.values(is_normalized)
            return vec
        else:
            return self.vector_value

    def get_feature_by_name(self, name):
        return self.features_dict[name]

    def __repr__(self):
        return str(self.vector(is_normalized=False))


class FeatureFamilyName(Enum):
    AOIR = "aoir"
    TIME_DISTANCES = "time_dist"


class FeatureFamily:
    def __init__(self, vvalues, fmin, fmax, name):
        self.fmax = fmax
        self.fmin = fmin
        self.vvalues = vvalues
        self.name = name

    def values(self, is_normalized=True):
        if is_normalized:
            try:
                return [min_max_normalizer(v, self.fmin, self.fmax) for v in self.vvalues]
            except:
                print(self.vvalues, self.fmin, self.fmax)
        else:
            return self.vvalues

    @staticmethod
    def time_distances(drone: Drone, targets):
        return [euclidean_distance(drone.coords, target.coords) / drone.speed for target in targets]

    @staticmethod
    def aoi_tol_ratio(drone: Drone, targets, next=0):
        res = []
        for target in targets:
            target: Target = target
            # set target need to 0 if this target is not necessary or is locked (OR)
            # -- target is locked from another drone (not this drone)
            # -- is inactive
            # TODO CHECK IGNORE CONDITION
            is_ignore_target = (target.lock is not None and target.lock != drone) or not target.active
            res_val = 0 if is_ignore_target else target.AOI_ratio(next)
            res.append(res_val)
        return res

    def __repr__(self):
        return str(self.vvalues)
