
from src.patrolling.patrolling_DQN import PatrollingDQN
from src.utilities.utilities import euclidean_distance
from src.utilities import config
import numpy as np
import time

# class Feature:
#     def __init__(self, name, values, normalization_factor, maxi, mini=0):
#         self.name = name
#         self.values = values
#         self.normalization_factor = normalization_factor
#         self.mini = mini
#         self.maxi = maxi


class State:
    def __init__(self, aois, time_distances, position, aoi_norm, time_norm, position_norm, is_final,
                 future_residuals, aoi_future_norm, is_flying, objective):

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

        self._objective = objective

    def objective(self, normalized=True):
        return self._objective if not normalized else self.normalize_feature(self._objective, self.position_norm+1, 0)

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

    def vector(self, normalized=True):
        """ NN INPUT """
        return list(self.residuals(normalized)) + list(self.time_distances(normalized)) + [self.is_flying(normalized)] + [self.objective(normalized)]
        # return [self.position()] + list(self.residuals()) + list(self.time_distances())

    def __repr__(self):
        return "res: {}\ndis: {}\nfly: {}\nobj: {}\n"\
            .format(self.residuals(), self.time_distances(), self.is_flying(False), self.objective(False))

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
        self.N_FEATURES = 2 * len(self.simulator.environment.targets) + 2

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
        self.AOI_NORM = 10 # self.simulator.duration_seconds() / min_threshold
        self.TIME_NORM = self.simulator.max_travel_time()
        self.ACTION_NORM = self.N_ACTIONS

    def get_current_residuals(self):
        """ max tra AOI / IDLENESS e 1 """
        MAX_TOL = 10
        return [min(target.aoi_idleness_ratio(), MAX_TOL)
                for target in self.simulator.environment.targets]

    def get_future_residuals(self):
        """ max tra (AOI + TRANSIT) / IDLENESS e 1 """
        fut = lambda i, t: (t.age_of_information() + self.get_current_time_distances()[i]) / t.maximum_tolerated_idleness
        return [min(fut(i, target), 1) for i, target in enumerate(self.simulator.environment.targets)]

    def get_current_time_distances(self):
        """ TIME of TRANSIT """
        return [euclidean_distance(self.drone.coords, target.coords)/self.drone.speed for target in self.drone.simulator.environment.targets]

    def evaluate_state(self):
        pa = self.previous_action if self.previous_action is not None else 0
        residuals = self.get_current_residuals()
        future_residuals = self.get_future_residuals()
        is_flying = self.drone.is_flying()
        objective = self.previous_action if self.drone.is_flying() else self.N_ACTIONS + 1
        distances = self.get_current_time_distances()

        state = State(residuals, distances, pa, self.AOI_NORM, self.TIME_NORM, self.ACTION_NORM,
                     False, future_residuals, self.AOI_FUTURE_NORM, is_flying, objective)
        return state

    def evaluate_reward(self, state):
        norm_residuals = state.residuals()
        dead_residuals_idx = [i for i, res in enumerate(state.residuals(False)) if res >= 1]
        dead_residuals = [norm_residuals[i] for i in dead_residuals_idx]
        # live_residuals = [res for res in state.residuals(False) if res < 1]

        # rew = (-sum(state.residuals())) if config.POSITIVE else sum([1-i for i in state.residuals()])
        # rew = rew / self.N_ACTIONS  # media sui target

        rew = -sum(dead_residuals) / self.N_ACTIONS if config.REW_MODE else -len(dead_residuals) / self.N_ACTIONS
        rew += rew if not state.is_final else config.PENALTY_ON_BS_EXPIRATION
        return rew

    def evaluate_is_final_state(self, s, a, s_prime):
        return s_prime.residuals(False)[0] >= 1  # or s.position() == s_prime.position()

    def invoke_train(self):
        if self.previous_state is None or self.previous_action is None:
            return 0, self.previous_epsilon, self.previous_loss, False, None, None

        self.DQN.n_decision_step += 1

        s = self.previous_state
        a = self.previous_action
        s_prime = self.evaluate_state()
        s_prime.is_final = self.evaluate_is_final_state(s, a, s_prime)
        r = self.evaluate_reward(s_prime)

        if config.LOG_STATE >= 0:
            self.log_transition(s, s_prime, a, r, every=config.LOG_STATE)

        # print(s.position(False), a, s_prime.position(False), r)
        self.previous_epsilon = self.DQN.decay()

        # Continuous Tasks: Reinforcement Learning tasks which are not made of episodes, but rather last forever.
        # This tasks have no terminal states. For simplicity, they are usually assumed to be made of one never-ending episode.
        self.previous_loss = self.DQN.train(previous_state=s.vector(),
                                            current_state=s_prime.vector(),
                                            action=a,
                                            reward=r,
                                            is_final=s_prime.is_final)

        if s_prime.is_final:
            self.simulator.environment.reset_drones_targets()
            self.previous_state = None
            self.previous_action = None
            self.policy_cycle = 0

        return r, self.previous_epsilon, self.DQN.current_loss, s_prime.is_final, s, s_prime

    def invoke_predict(self, state):
        if state is None:
            state = self.evaluate_state()

        action_index, q = self.DQN.predict(state.vector())
        if bool(state.is_flying()):
            action_index = state.objective(False)

        self.previous_state = state
        self.previous_action = action_index
        return action_index, q[0]

    def log_transition(self, s, s_prime, a, r, every=1):
        print(s)
        print(s_prime)
        print(a, r)
        print("---")
        time.sleep(every)
