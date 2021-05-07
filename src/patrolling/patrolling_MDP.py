
from src.patrolling.patrolling_DQN import PatrollingDQN
from src.utilities.utilities import euclidean_distance
from src.utilities import config
import numpy as np


# class Feature:
#     def __init__(self, name, values, normalization_factor, maxi, mini=0):
#         self.name = name
#         self.values = values
#         self.normalization_factor = normalization_factor
#         self.mini = mini
#         self.maxi = maxi


class State:
    def __init__(self, aois, time_distances, position, aoi_norm, time_norm, position_norm, is_final,
                 future_residuals, aoi_future_norm, is_flying):

        self._future_residuals: list = future_residuals
        self._residuals: list = aois
        self._time_distances: list = time_distances
        self._position = position
        self.is_final = is_final
        self._is_flying = is_flying

        self.aoi_future_norm = aoi_future_norm
        self.aoi_norm = aoi_norm
        self.time_norm = time_norm
        self.position_norm = position_norm

    def is_flying(self, normalized=True):
        return self._is_flying if not normalized else int(self._is_flying)

    def future_residuals(self, normalized=True):
        return self._future_residuals if not normalized else self.normalize_feature(self._future_residuals, self.aoi_future_norm, 0)

    def residuals(self, normalized=True):
        return self._residuals if not normalized else self.normalize_feature(self._residuals, self.aoi_norm, 0)

    def time_distances(self, normalized=True):
        return self._time_distances if not normalized else self.normalize_feature(self._time_distances, self.time_norm, 0)

    def position(self, normalized=True):
        return self._position if not normalized else self.normalize_feature(self._position, self.position_norm, 0)

    def normalized_vector(self):
        """ NN INPUT """
        return list(self.residuals()) + list(self.future_residuals())
        # return [self.position()] + list(self.residuals()) + list(self.time_distances())

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
        self.previous_epsilon = 1
        self.previous_loss = None

        self.com_rewards = 0
        self.policy_cycle = 0

        self.N_ACTIONS = len(self.simulator.environment.targets)
        self.N_FEATURES = 2 * len(self.simulator.environment.targets)

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
        self.AOI_FUTURE_NORM = 1  # self.simulator.duration_seconds() / min_threshold
        self.AOI_NORM = 1  # self.simulator.duration_seconds() / min_threshold
        self.TIME_NORM = None if config.RELATIVE else self.simulator.max_travel_time()
        self.ACTION_NORM = self.N_ACTIONS

    def get_current_residuals(self):
        """ max tra AOI / IDLENESS e 1 """
        return [min(target.aoi_idleness_ratio(), 1) for target in self.simulator.environment.targets]

    def get_future_residuals(self):
        """ max tra (AOI + TRANSIT) / IDLENESS e 1 """
        fut = lambda i, t: (t.age_of_information() + self.get_current_time_distances()[i]) / t.maximum_tolerated_idleness
        return [min(fut(i, target), 1) for i, target in enumerate(self.simulator.environment.targets)]

    def get_current_time_distances(self):
        """ TIME of TRANSIT """
        return [euclidean_distance(self.drone.coords, target.coords)/self.drone.speed for target in self.drone.simulator.environment.targets]

    def get_is_flying(self):
        return self.drone.is_flying()

    def evaluate_state(self):
        pa = self.previous_action if self.previous_action is not None else 0
        residuals = self.get_current_residuals()
        future_residuals = self.get_future_residuals()
        is_flying = self.get_is_flying()

        # distances = self.get_current_time_distances()
        # thresholds = np.asarray([target.maximum_tolerated_idleness for target in self.simulator.environment.targets])
        # distances2 = distances / (thresholds if self.TIME_NORM is None else 1)
        # print(distances)
        # print(thresholds)
        # print(distances2)
        # print()

        return State(residuals, future_residuals, pa, self.AOI_NORM, self.TIME_NORM, self.ACTION_NORM, False, future_residuals, self.AOI_FUTURE_NORM, is_flying)

    def evaluate_reward(self, state):
        # dead_residuals = [res for res in state.residuals(False) if res >= 1]
        # live_residuals = [res for res in state.residuals(False) if res < 1]

        rew = (-sum(state.residuals())) if not config.POSITIVE else sum([1-i for i in state.residuals()])
        rew = rew / self.N_ACTIONS  # media sui target

        # rew = (len(live_residuals) if config.POSITIVE else (- len(dead_residuals))) / self.N_ACTIONS
        rew = rew if not state.is_final else -5
        return rew

    def evaluate_is_final_state(self, s, a, s_prime):
        return s_prime.future_residuals()[0] >= 1 # or s.position() == s_prime.position()

    def invoke_train(self):
        if self.previous_state is None or self.previous_action is None:
            # print("s", None)
            # print()
            return 0, self.previous_epsilon, self.previous_loss, False, None

        self.DQN.n_decision_step += 1

        s = self.previous_state
        a = self.previous_action
        s_prime = self.evaluate_state()
        s_prime.is_final = self.evaluate_is_final_state(s, a, s_prime)
        r = self.evaluate_reward(s_prime)

        # print("s", s.position(False), s.is_final)
        # print("a", a)
        # print("s'", s_prime.position(False), s_prime.is_final)
        # print("r", r)
        # print(self.previous_loss)
        # print()

        # print(s.position(False), a, s_prime.position(False), r)
        self.previous_epsilon = self.DQN.decay()

        # Continuous Tasks: Reinforcement Learning tasks which are not made of episodes, but rather last forever.
        # This tasks have no terminal states. For simplicity, they are usually assumed to be made of one never-ending episode.
        self.previous_loss = self.DQN.train(previous_state=s.normalized_vector(),
                                            current_state=s_prime.normalized_vector(),
                                            action=a,
                                            reward=r,
                                            is_final=s_prime.is_final)

        if s_prime.is_final:
            self.simulator.environment.reset_drones_targets()
            self.previous_state = None
            self.previous_action = None
            self.policy_cycle = 0

        return r, self.previous_epsilon, self.DQN.current_loss, s_prime.is_final, s_prime

    def invoke_predict(self, state, force_action=None):
        if state is None:
            state = self.evaluate_state()

        action_index = self.DQN.predict(state.normalized_vector()) if force_action is None else force_action
        self.previous_state = state
        self.previous_action = action_index
        self.policy_cycle += 1
        return action_index


