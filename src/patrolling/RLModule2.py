
from src.utilities.utilities import min_max_normalizer, euclidean_distance
from enum import Enum

from src.world_entities.drone import Drone
from src.world_entities.target import Target


class RLModule:

    def __init__(self, simulator):
        self.sim = simulator
        self.cf = self.sim.cf

    def train_DQN(self):
        pass

    def test_DQN(self):
        pass

    def state(self, drone):
        distances = FeatureFamily.time_distances(drone, self.sim.environment.targets)
        distances = FeatureFamily(distances, 0, self.cf.max_time_distance())

        aoir = FeatureFamily.aoi_tol_ratio(drone, self.sim.environment.targets)
        aoir = FeatureFamily(aoir, 0, 1)

        features = [distances, aoir]
        state = State(features)
        return state

    def reward(self):
        pass

    def action(self):
        return


class State:
    def __init__(self, features: list):
        self.features = features
        self.vector_value = None

    def vector(self, is_normalized=True):
        if self.vector_value is None:
            vec = []
            for f in self.features:
                vec += f.values(is_normalized)
            return vec
        else:
            return self.vector_value

    def __repr__(self):
        return str(self.vector(is_normalized=False))


class FeatureFamily:
    def __init__(self, vvalues, fmax, fmin):
        self.fmax = fmax
        self.fmin = fmin
        self.vvalues = vvalues

    def values(self, is_normalized=True):
        if is_normalized:
            return [min_max_normalizer(v, self.fmin, self.fmax) for v in self.vvalues]
        else:
            return self.vvalues

    @staticmethod
    def time_distances(drone:Drone, targets):
        return [euclidean_distance(drone.coords, target.coords) / drone.speed for target in targets]

    @staticmethod
    def aoi_tol_ratio(drone:Drone, targets, next=0):
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
