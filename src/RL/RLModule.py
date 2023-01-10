
from src.RL.DQNTraining import PatrollingDQN
from src.RL.RLSate import State, FeatureFamily, FeatureFamilyName
import numpy as np
from src.world_entities.drone import Drone
from src.RL.RLRewards import reward_map


class RLModule:

    def __init__(self, simulator):
        self.sim = simulator
        self.cf = self.sim.cf

        self.n_actions = self.sim.cf.TARGETS_NUMBER + 1
        self.n_state_features = 2 * self.n_actions
        self.dqn_mod = PatrollingDQN(self.cf, self.sim, self.n_actions, self.n_state_features)

    def query_model(self, drone: Drone, is_exploit=False):
        s_prime = self.state(drone)
        a_prime = self.action(s_prime.vector(), is_exploit)
        s_prime.is_final = self.sim.is_final_state()

        if is_exploit:
            return a_prime

        # s_prev -> a_prev -> s_prime (from here will execute, a_prime)
        s_prev = drone.prev_state
        s_prev_vec = s_prev.vector() if s_prev is not None else None
        a_prev = drone.prev_action

        r = self.reward(s_prev, s_prime, a_prime)

        # s_prev, s_prime, a_prev, r
        self.dqn_mod.train(s_prev_vec, s_prime.vector(), a_prev, r, not s_prime.is_final)

        drone.prev_state = s_prime
        drone.prev_action = a_prime
        return a_prime

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

    def reward(self, state_prev: State, state_prime: State, a_prev: int):
        return reward_map(self.cf.REWARD_TYPE, self.sim, state_prev, state_prime, a_prev)

    def action(self, state: State, is_exploit=False):
        return self.dqn_mod.predict(state, is_allowed_explore=not is_exploit)
