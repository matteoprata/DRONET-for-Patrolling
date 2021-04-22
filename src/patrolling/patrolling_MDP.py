
from src.patrolling.patrolling_DQN import PatrollingDQN
from src.utilities.utilities import euclidean_distance
from src.utilities import config
import numpy as np


class State:
    def __init__(self, aois, time_distances, position, aoi_norm, time_norm, position_norm, is_final):
        self._residuals: list = aois
        self._time_distances: list = time_distances
        self._position = position
        self.is_final = is_final

        self.aoi_norm = aoi_norm
        self.time_norm = time_norm
        self.position_norm = position_norm

    def residuals(self, normalized=True):
        return self._residuals if not normalized else self.normalize_feature(self._residuals, self.aoi_norm)

    def time_distances(self, normalized=True):
        return self._time_distances if not normalized else self.normalize_feature(self._time_distances, self.time_norm)

    def position(self, normalized=True):
        return self._position if not normalized else self.normalize_feature(self._position, self.position_norm)

    def normalized_vector(self):
        """ NN INPUT """
        return [self.position()] + list(self.residuals()) + list(self.time_distances())

    # def __repr__(self):
    #     return "{}\n{}\n{}".format(self.round_feature_vector(self._residuals, 3), self.round_feature_vector(self._time_distances, 3), self._position)

    @staticmethod
    def normalize_feature(feature, normal_factor):
        return np.array(feature) * 1 / normal_factor

    @staticmethod
    def round_feature_vector(feature, rounding_digit):
        return [round(i, rounding_digit) for i in feature]


class RLModule:
    def __init__(self, drone):
        self.drone = drone
        self.previous_state = None
        self.previous_action = None
        self.com_rewards = 0

        self.N_ACTIONS = len(self.drone.simulator.environment.targets)
        self.N_FEATURES = 2*len(self.drone.simulator.environment.targets)+1

        self.DQN = PatrollingDQN(pretrained_model_path=config.RL_MODEL,
                                 n_actions=self.N_ACTIONS,
                                 n_features=self.N_FEATURES,
                                 simulator=self.drone.simulator,
                                 metrics=self.drone.simulator.metrics,
                                 load_model=config.PRE_TRAINED
                                 )

        self.AOI_NORM = self.drone.simulator.sim_duration_ts * self.drone.simulator.ts_duration_sec
        self.TIME_NORM = self.drone.simulator.max_travel_time()
        self.ACTION_NORM = (self.drone.simulator.n_targets + self.drone.simulator.n_drones)

    def get_current_residuals(self):
        return [target.residual_of_information() for target in self.drone.simulator.environment.targets]

    def get_current_time_distances(self):
        return [euclidean_distance(self.drone.coords, target.coords)/self.drone.speed for target in self.drone.simulator.environment.targets]

    def evaluate_state(self):
        pa = self.previous_action if self.previous_action is not None else 0

        AOI_NORM = self.drone.simulator.sim_duration_ts * self.drone.simulator.ts_duration_sec
        TIME_NORM = self.drone.simulator.max_travel_time()
        ACTION_NORM = (self.drone.simulator.n_targets + self.drone.simulator.n_drones)

        residuals = self.get_current_residuals()
        is_final = residuals[0] < 0
        return State(residuals, self.get_current_time_distances(), pa, AOI_NORM, TIME_NORM, ACTION_NORM, is_final)

    def evaluate_reward(self, state):
        zero_residuals = [res for res in state.residuals(False) if res < 0]
        rew = - len(zero_residuals)/self.N_ACTIONS if len(zero_residuals) > 0 else 2  # len / min / mean
        rew = rew if not state.is_final else -2
        return rew

    def invoke_train(self):
        if self.previous_state is None or self.previous_action is None:
            return

        self.DQN.n_decision_step += 1

        s = self.previous_state
        a = self.previous_action
        s_prime = self.evaluate_state()
        r = self.evaluate_reward(s_prime)

        self.drone.simulator.metrics.cum_reward += r
        self.drone.simulator.metrics.cum_residuals += np.average(s_prime.residuals(False))

        if s_prime.is_final:
            self.drone.simulator.environment.reset_drones_targets()

        # Continuous Tasks: Reinforcement Learning tasks which are not made of episodes, but rather last forever.
        # This tasks have no terminal states. For simplicity, they are usually assumed to be made of one never-ending episode.
        self.DQN.train(previous_state=s.normalized_vector(),
                       current_state=s_prime.normalized_vector(),
                       action=a,
                       reward=r,
                       is_final=s_prime.is_final)

    def invoke_predict(self):
        s = self.evaluate_state()
        action_index = self.DQN.predict(s.normalized_vector())

        self.previous_state = s
        self.previous_action = action_index
        return action_index


