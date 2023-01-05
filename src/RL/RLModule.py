
from src.RL.DQNTraining import PatrollingDQN
from src.RL.RLSate import State, FeatureFamily, FeatureFamilyName
import numpy as np
from src.utilities.utilities import min_max_normalizer
from src.world_entities.drone import Drone


class RLModule:

    def __init__(self, simulator):
        self.sim = simulator
        self.cf = self.sim.cf

        n_actions = self.sim.cf.TARGETS_NUMBER + 1
        n_state_features = 2 * n_actions
        self.dqn_mod = PatrollingDQN(self.cf, self.sim, n_actions, n_state_features)

    def query_model(self, drone: Drone, is_exploit=False):
        s_prime = self.state(drone)
        a_prime = self.action(s_prime.vector(), is_exploit)

        if is_exploit:
            return a_prime

        r = self.reward(s_prime)

        # s_prev -> a_prev -> s_prime (from here will execute, a_prime)
        s_prev = drone.prev_state
        s_prev_vec = s_prev.vector() if s_prev is not None else None
        a_prev = drone.prev_action

        # s_prev, s_prime, a_prev, r
        self.dqn_mod.train(s_prev_vec, s_prime.vector(), a_prev, r)

        drone.prev_state = s_prime
        drone.prev_action = a_prime
        return a_prime

    # ----> MDP ahead < ----

    # RESEARCH HERE
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

    # RESEARCH HERE
    def reward(self, state: State):
        # r1 = - np.sum(state.get_feature_by_name(FeatureFamilyName.AOIR).values()) # - SUM AOIR  [50%, 150%] -> 200%

        r2_vals = np.array(state.get_feature_by_name(FeatureFamilyName.AOIR).values(is_normalized=False))  # - SUM AOIR > 100%  [50%, 150%] -> 150%
        r2 = - np.sum(r2_vals[r2_vals >= 1])
        r2_norm = min_max_normalizer(r2, self.sim.rmin, 0, -1, 0, soft=True)
        return r2_norm

    def action(self, state: State, is_exploit=False):
        return self.dqn_mod.predict(state, is_allowed_explore=not is_exploit)
