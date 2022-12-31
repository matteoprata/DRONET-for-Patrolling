
from src.patrolling.DQN_module2 import PatrollingDQN
from src.patrolling.State2 import State, FeatureFamily, FeatureFamilyName
import numpy as np

from src.world_entities.drone import Drone

class RLModule:

    def __init__(self, simulator):
        self.sim = simulator
        self.cf = self.sim.cf

        n_actions = len(self.sim.environment.targets)
        n_state_features = 2 * n_actions
        self.dqn_mod = PatrollingDQN(self.cf, self.sim, n_actions, n_state_features)

    def query_model(self, drone: Drone, is_exploit=False):
        s_prime = self.state(drone)
        a = self.action(s_prime.vector(), is_exploit)

        if is_exploit:
            return a

        r = self.reward(s_prime)
        s = drone.prev_state
        s_vec = s.vector() if s is not None else None
        drone.prev_state = s_prime

        self.dqn_mod.train(s_vec, s_prime.vector(), a, r)
        return a

    # ----> MDP ahead < ----

    def state(self, drone) -> State:
        # n targets features
        distances = FeatureFamily.time_distances(drone, self.sim.environment.targets)
        distances = FeatureFamily(distances, 0, self.cf.max_time_distance(), FeatureFamilyName.TIME_DISTANCES)

        # n targets features
        aoir = FeatureFamily.aoi_tol_ratio(drone, self.sim.environment.targets)
        aoir = FeatureFamily(aoir, 0, self.cf.max_times_violation(), FeatureFamilyName.AOIR)

        features = [distances, aoir]
        state = State(features)
        return state

    def reward(self, state: State):
        return - np.sum(state.get_feature_by_name(FeatureFamilyName.AOIR).values())

    def action(self, state: State, is_exploit=False):
        return self.dqn_mod.predict(state, is_allowed_explore=not is_exploit)
