
from src.patrolling.patrolling_DQN_TORCH import PatrollingDQN
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
    def __init__(self, residuals, time_distances, position, aoi_norm, time_norm, position_norm, is_final, is_flying, objective):

        self._residuals: list = residuals
        self._time_distances: list = time_distances

        self.aoi_norm = aoi_norm
        self.time_norm = time_norm

        # UNUSED from here
        self._position = position
        self.is_final = is_final
        self._is_flying = is_flying
        self.position_norm = position_norm
        self._objective = objective

    def objective(self, normalized=True):
        return self._objective if not normalized else self.normalize_feature(self._objective, self.position_norm+1, 0)

    def is_flying(self, normalized=True):
        return self._is_flying if not normalized else int(self._is_flying)

    def residuals(self, normalized=True):
        return self._residuals if not normalized else self.normalize_feature(self._residuals, self.aoi_norm, 0)

    def time_distances(self, normalized=True):
        return self._time_distances if not normalized else self.normalize_feature(self._time_distances, self.time_norm, 0)

    def position(self, normalized=True):
        return self._position if not normalized else self.normalize_feature(self._position, self.position_norm, 0)

    def vector(self, normalized=True, rounded=False):
        """ NN INPUT """
        if not rounded:
            return list(self.residuals(normalized)) + list(self.time_distances(normalized))
        else:
            return [round(i, 2) for i in list(self.residuals(normalized))] + \
                   [round(i, 2) for i in list(self.time_distances(normalized))]

    def __repr__(self):
        return "res: {}\ndis: {}\n".format(self.residuals(), self.time_distances()) #self.is_flying(False), self.objective(False))

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

        self.prev_learning_tuple = None
        self.N_ACTIONS = len(self.simulator.environment.targets)
        self.N_FEATURES = 2 * len(self.simulator.environment.targets)
        self.MAX_RES_PRECISION = 10  # above this we are not interested on how much the

        self.DQN = PatrollingDQN(n_actions=self.N_ACTIONS,
                                 n_features=self.N_FEATURES,
                                 simulator=self.simulator,
                                 metrics=self.simulator.metrics,
                                 lr =                       self.simulator.learning["learning_rate"],
                                 batch_size =               self.simulator.learning["batch_size"],
                                 is_load_model=               self.simulator.learning["is_pretrained"],
                                 pretrained_model_path =    self.simulator.learning["model_name"],
                                 discount_factor =          self.simulator.learning["discount_factor"],
                                 replay_memory_depth =      self.simulator.learning["replay_memory_depth"],
                                 swap_models_every_decision=self.simulator.learning["swap_models_every_decision"],
                                 )

        self.AOI_NORM = self.MAX_RES_PRECISION  # self.simulator.duration_seconds() / min_threshold
        self.TIME_NORM = self.simulator.max_travel_time()

        # min_threshold = min([t.maximum_tolerated_idleness for t in self.simulator.environment.targets])
        # self.AOI_FUTURE_NORM = 1  # self.simulator.duration_seconds() / min_threshold
        # self.ACTION_NORM = self.N_ACTIONS

    def get_current_residuals(self, next=0):
        return [min(target.aoi_idleness_ratio(next), self.MAX_RES_PRECISION) for target in self.simulator.environment.targets]

    def get_current_time_distances(self):
        """ TIME of TRANSIT """
        return [euclidean_distance(self.drone.coords, target.coords)/self.drone.speed for target in self.drone.simulator.environment.targets]

    def evaluate_state(self):
        # pa = self.previous_action if self.previous_action is not None else 0
        # is_flying = self.drone.is_flying()
        # objective = self.previous_action if self.drone.is_flying() else self.N_ACTIONS + 1

        # - - # - - # - - # - - # - - # - - # - - # - - # - - #
        distances = self.get_current_time_distances()
        residuals = self.get_current_residuals()

        state = State(residuals, distances, None, self.AOI_NORM, self.TIME_NORM, None, False, None, None)
        return state

    def evaluate_reward(self, s, a, s_prime):
        # REW = 0
        # EMPHASYZE = 0
        #
        # time_dist_to_a = s.time_distances(False)[a]
        # n_steps = max(int(time_dist_to_a/config.DELTA_DEC), 1)
        #
        # for step in range(n_steps):
        #     TIME = self.simulator.current_second() - (config.DELTA_DEC * step)
        #
        #     for target in self.simulator.environment.targets:
        #         LAST_VISIT = target.last_visit_ts * self.simulator.ts_duration_sec
        #         residual = (TIME - LAST_VISIT) / target.maximum_tolerated_idleness
        #
        #         # print(target.identifier, a, time_dist_to_a, LAST_VISIT, residual)
        #         residual = min(residual, self.MAX_RES_PRECISION)  # 10
        #         REW += - residual if residual >= 1 else 0
        #
        # norm_factor_rew = self.N_ACTIONS * self.MAX_RES_PRECISION * int(self.simulator.max_travel_time() / config.DELTA_DEC)
        #
        # # print(REW, REW / norm_factor_rew)
        # REW = REW / norm_factor_rew + EMPHASYZE
        #
        # REW += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        # return REW

        rew = - sum([i for i in s_prime.residuals(False) if i >= 1]) / (self.MAX_RES_PRECISION * self.N_ACTIONS)
        rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        return rew


    def evaluate_is_final_state(self, s, a, s_prime):
        """ The residual of the base station is >= 1, i.e. it is expired. """
        return s_prime.residuals(False)[0] >= 1  # or s.position() == s_prime.position()

    def invoke_train(self):
        if self.previous_state is None or self.previous_action is None:
            return 0, self.previous_epsilon, self.previous_loss, False, None, None

        self.DQN.n_decision_step += 1

        s = self.previous_state
        a = self.previous_action
        s_prime = self.evaluate_state()
        s_prime.is_final = self.evaluate_is_final_state(s, a, s_prime)
        r = self.evaluate_reward(s, a, s_prime)

        if not self.drone.is_flying():
            s_prime._residuals[a] = 0  # set to 0 the residual of the just visited target (it will be reset later from drone.py)

        self.prev_learning_tuple = s.vector(False, True), a, s_prime.vector(False, True), r

        if self.simulator.log_state >= 0:
            self.log_transition(s, s_prime, a, r, every=self.simulator.log_state)

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
            self.simulator.environment.reset_drones_targets(False)
            self.reset_MDP()

        return r, self.previous_epsilon, self.DQN.current_loss, s_prime.is_final, s, s_prime

    def invoke_predict(self, state):
        if state is None:
            state = self.evaluate_state()

        action_index, q = self.DQN.predict(state.vector())

        if self.drone.is_flying():
            action_index = self.previous_action

        self.previous_state = state
        self.previous_action = action_index
        return action_index, q[0]

    def reset_MDP(self):
        self.previous_state = None
        self.previous_action = None

    def log_transition(self, s, s_prime, a, r, every=1):
        print(s.vector(False))
        print(s_prime.vector(False))
        print(a, r)
        print("---")
        time.sleep(every)
