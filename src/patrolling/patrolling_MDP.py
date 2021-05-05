
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
        return self._residuals if not normalized else self.normalize_feature(self._residuals, self.aoi_norm, 0)

    def time_distances(self, normalized=True):
        return self._time_distances if not normalized else self.normalize_feature(self._time_distances, self.time_norm, 0)

    def position(self, normalized=True):
        return self._position if not normalized else self.normalize_feature(self._position, self.position_norm, 0)

    def normalized_vector(self):
        """ NN INPUT """
        return [self.position()] + list(self.residuals()) + list(self.time_distances())

    def __repr__(self):
        return "{}\n{}\n{}".format(self.position(), self.residuals(), self.time_distances())

    @staticmethod
    def normalize_feature(feature, maxi, mini):
        return ((np.asarray(feature) - mini) / (maxi - mini)) if not (maxi is None or mini is None) else np.asarray(feature)

    @staticmethod
    def round_feature_vector(feature, rounding_digit):
        return [round(i, rounding_digit) for i in feature]


class RLModule:
    def __init__(self, drone):
        self.drone = drone
        self.simulator = self.drone.simulator

        self.previous_state = None
        self.previous_action = None
        self.com_rewards = 0

        self.N_ACTIONS = len(self.simulator.environment.targets)
        self.N_FEATURES = 2 * len(self.simulator.environment.targets) + 1

        self.DQN = PatrollingDQN(n_actions=self.N_ACTIONS,
                                 n_features=self.N_FEATURES,
                                 simulator=self.simulator,
                                 metrics=self.simulator.metrics,
                                 beta =                       self.simulator.learning["beta"],
                                 lr =                         self.simulator.learning["learning_rate"],
                                 batch_size =                 self.simulator.learning["batch_size"],
                                 load_model =                 self.simulator.learning["is_pretrained"],
                                 epsilon_decay =              self.simulator.learning["epsilon_decay"],
                                 pretrained_model_path =      self.simulator.learning["model_name"],
                                 discount_factor =            self.simulator.learning["discount_factor"],
                                 replay_memory_depth =        self.simulator.learning["replay_memory_depth"],
                                 swap_models_every_decision = self.simulator.learning["swap_models_every_decision"],
                                 )

        min_threshold = min([t.maximum_tolerated_idleness for t in self.simulator.environment.targets])
        self.AOI_NORM = 1  # self.simulator.duration_seconds() / min_threshold
        self.TIME_NORM = None if config.RELATIVE else self.simulator.max_travel_time()
        self.ACTION_NORM = self.N_ACTIONS

    def get_current_residuals(self):
        return [min(target.aoi_idleness_ratio(), 1) for target in self.simulator.environment.targets]

    def get_current_time_distances(self):
        return [euclidean_distance(self.drone.coords, target.coords)/self.drone.speed for target in self.drone.simulator.environment.targets]

    def evaluate_state(self):
        pa = self.previous_action if self.previous_action is not None else 0
        residuals = self.get_current_residuals()
        distances = np.asarray(self.get_current_time_distances())
        thresholds = np.asarray([target.maximum_tolerated_idleness for target in self.simulator.environment.targets])
        distances = distances / (thresholds if self.TIME_NORM is None else 1)
        return State(residuals, distances, pa, self.AOI_NORM, self.TIME_NORM, self.ACTION_NORM, False)

    def evaluate_reward(self, state, action):
        live_residuals = [res for res in state.residuals(False) if res < 1]
        dead_residuals = [res for res in state.residuals(False) if res >= 1]

        rew = (len(live_residuals) if config.POSITIVE else (- len(dead_residuals))) / self.N_ACTIONS
        rew += 0 if not state.is_final else -2
        return rew

    def evaluate_is_final_state(self, s, a, s_prime):
        return s_prime.residuals()[0] >= 1 or s.position() == s_prime.position()

    def invoke_train(self):
        if self.previous_state is None or self.previous_action is None:
            return 0, 1, 0, False, None

        self.DQN.n_decision_step += 1

        s = self.previous_state
        a = self.previous_action
        s_prime = self.evaluate_state()
        s_prime.is_final = self.evaluate_is_final_state(s, a, s_prime)

        r = self.evaluate_reward(s_prime, a)

        if s_prime.is_final:
            self.simulator.environment.reset_drones_targets()

        # Continuous Tasks: Reinforcement Learning tasks which are not made of episodes, but rather last forever.
        # This tasks have no terminal states. For simplicity, they are usually assumed to be made of one never-ending episode.
        self.DQN.train(previous_state=s.normalized_vector(),
                       current_state=s_prime.normalized_vector(),
                       action=a,
                       reward=r,
                       is_final=s_prime.is_final)

        return r, self.DQN.decay(), self.DQN.current_loss, s_prime.is_final, s_prime

    def invoke_predict(self, state):
        if state is None:
            state = self.evaluate_state()

        action_index = self.DQN.predict(state.normalized_vector())
        self.previous_state = state
        self.previous_action = action_index
        return action_index


