
from src.patrolling.patrolling_DQN_TORCH import PatrollingDQN
from src.utilities.utilities import euclidean_distance, min_max_normalizer
from src.utilities import config
import numpy as np
import time


class State:
    def __init__(self, aoi_idleness_ratio, time_distances, position, aoi_norm, time_norm,
                 position_norm, is_final, is_flying, objective, closests, actions_past):

        self._aoi_idleness_ratio: list = aoi_idleness_ratio
        self._time_distances: list = time_distances
        self._closests: list = closests
        self._actions_past: list = actions_past

        self.aoi_norm = aoi_norm
        self.time_norm = time_norm

        # UNUSED from here
        self._position = position
        self.is_final = is_final
        self._is_flying = is_flying
        self.position_norm = position_norm
        self._objective = objective

    def actions_past(self, normalized=True):
        return self._actions_past if not normalized else min_max_normalizer(self._actions_past, 0, self.position_norm)

    # def is_flying(self, normalized=True):
    #     return self._is_flying if not normalized else int(self._is_flying)

    def aoi_idleness_ratio(self, normalized=True):
        return self._aoi_idleness_ratio if not normalized else min_max_normalizer(self._aoi_idleness_ratio, 0, self.aoi_norm)

    def time_distances(self, normalized=True):
        return self._time_distances if not normalized else min_max_normalizer(self._time_distances, 0, self.time_norm)

    def position(self, normalized=True):
        return self._position if not normalized else min_max_normalizer(self._position, 0, self.position_norm)

    def closests(self, normalized=True):
        return self._closests if not normalized else min_max_normalizer(self._closests, 0, self.time_norm)

    def vector(self, normalized=True, rounded=False):
        """ NN INPUT """
        if not rounded:
            return list(self.aoi_idleness_ratio(normalized)) + list(self.time_distances(normalized)) + list(self.closests(normalized)) + list(self.actions_past(normalized))
        else:
            return [round(i, 2) for i in list(self.aoi_idleness_ratio(normalized))] + \
                   [round(i, 2) for i in list(self.time_distances(normalized))] + \
                   [round(i, 2) for i in list(self.closests(normalized))] + \
                   [round(i, 2) for i in list(self.actions_past(normalized))]

    def __repr__(self):
        return "res: {}\ndis: {}\nclo: {}\n".format(self.aoi_idleness_ratio(), self.time_distances(), self.closests()) #self.is_flying(False), self.objective(False))

    @staticmethod
    def round_feature_vector(feature, rounding_digit):
        return [round(i, rounding_digit) for i in feature]


class RLModule:
    def __init__(self, environment):
        self.environment = environment
        self.simulator = self.environment.simulator

        self.previous_epsilon = 1
        self.previous_loss = None

        self.N_ACTIONS = len(self.environment.targets)
        self.N_FEATURES = 3 * len(self.environment.targets) + len(self.environment.drones)
        self.TARGET_VIOLATION_FACTOR = config.TARGET_VIOLATION_FACTOR  # above this we are not interested on how much the

        self.DQN = PatrollingDQN(n_actions=self.N_ACTIONS,
                                 n_features=self.N_FEATURES,
                                 n_hidden_neurons_lv1=self.simulator.learning["n_hidden_neurons_lv1"],
                                 n_hidden_neurons_lv2=self.simulator.learning["n_hidden_neurons_lv2"],
                                 n_hidden_neurons_lv3=self.simulator.learning["n_hidden_neurons_lv3"],
                                 simulator=self.simulator,
                                 metrics=self.simulator.metrics,
                                 lr =                       self.simulator.learning["learning_rate"],
                                 batch_size =               self.simulator.learning["batch_size"],
                                 is_load_model=             self.simulator.learning["is_pretrained"],
                                 pretrained_model_path =    self.simulator.learning["model_name"],
                                 discount_factor =          self.simulator.learning["discount_factor"],
                                 replay_memory_depth =      self.simulator.learning["replay_memory_depth"],
                                 swap_models_every_decision=self.simulator.learning["swap_models_every_decision"],
                                 )

        self.AOI_NORM = self.TARGET_VIOLATION_FACTOR  # self.simulator.duration_seconds() / min_threshold
        self.TIME_NORM = self.simulator.max_travel_time()

        # min_threshold = min([t.maximum_tolerated_idleness for t in self.simulator.environment.targets])
        # self.AOI_FUTURE_NORM = 1  # self.simulator.duration_seconds() / min_threshold
        # self.ACTION_NORM = self.N_ACTIONS

    def get_current_aoi_idleness_ratio(self, drone, next=0):
        res = []
        for target in self.simulator.environment.targets:
            # set target need to 0 if this target is not necessary or is locked (OR)
            # -- target is locked from another drone (not this drone)
            # -- is inactive
            is_ignore_target = self.simulator.learning["is_pretrained"] and ((target.lock is not None and target.lock != drone) or not target.active)
            res_val = 0 if is_ignore_target else min(target.aoi_idleness_ratio(next), self.TARGET_VIOLATION_FACTOR)
            res.append(res_val)
        return res

    def get_current_time_distances(self, drone):
        """ TIME of TRANSIT """
        return [euclidean_distance(drone.coords, target.coords)/drone.speed for target in drone.simulator.environment.targets]

    def get_targets_closest_drone(self):
        """ The shortest distance drone <> target for every target.  """
        min_distances = []
        for tar in self.environment.targets:
            clo_time = np.inf
            for dr in self.environment.drones:
                dis = euclidean_distance(dr.coords, tar.coords)
                tem = dis / dr.speed
                clo_time = tem if tem < clo_time else clo_time
            min_distances.append(clo_time)
        return min_distances

    def prev_actions(self):
        return [0] * len(self.environment.drones) if self.environment.read_previous_actions_drones[0] is None else self.environment.read_previous_actions_drones

    def evaluate_state(self, drone):
        # - - # - - # - - # - - # - - # - - # - - # - - # - - #
        distances = self.get_current_time_distances(drone)
        residuals = self.get_current_aoi_idleness_ratio(drone)
        closests = self.get_targets_closest_drone()
        actions_past = self.prev_actions()

        state = State(residuals, distances, None, self.AOI_NORM, self.TIME_NORM, self.N_ACTIONS, False, None, None, closests, actions_past)
        return state

    def __rew_on_flight(self, s, a, s_prime, drone):
        # if config.IS_RESIDUAL_REWARD:
        #     sum_exp_res = sum([max(1-i, -self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False)])
        #     rew = sum_exp_res + (self.simulator.petnalty_on_bs_expiration if s_prime.is_final else 0)
        # else:
        #     sum_exp_res = - sum([min(i, self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False) if i >= 1])
        #     rew = sum_exp_res + (self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0)
        #     rew = min_max_normalizer(rew,
        #                              startLB=(- (self.TARGET_VIOLATION_FACTOR * self.N_ACTIONS) + self.simulator.penalty_on_bs_expiration),
        #                              startUB=0,
        #                              endLB=-1,
        #                              endUB=0)
        FRAC_MOV = - 0.01
        rew = - max([min(i, self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False)])
        rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        rew += FRAC_MOV

        rew = min_max_normalizer(rew,
                                 startUB=0,
                                 startLB=-(self.TARGET_VIOLATION_FACTOR + FRAC_MOV + self.N_ACTIONS),
                                 endUB=0,
                                 endLB=-1)

        return rew


    def __rew_on_target(self, s, a, s_prime):
        pass  # old

        # rew = 0
        #
        # time_dist_to_a = s.time_distances(False)[a]
        # n_steps = max(int(time_dist_to_a/config.DELTA_DEC), 1)
        #
        # for step in range(n_steps):
        #     TIME = self.simulator.current_second() - (config.DELTA_DEC * step)
        #
        #     for target in self.simulator.environment.targets:
        #         LAST_VISIT = target.last_visit_ts * self.simulator.ts_duration_sec
        #         if config.IS_RESIDUAL_REWARD:
        #             residual = 1 - (TIME - LAST_VISIT) / target.maximum_tolerated_idleness
        #             residual = max(residual, -self.TARGET_VIOLATION_FACTOR)  # 10
        #             rew += residual
        #         else:
        #             residual = (TIME - LAST_VISIT) / target.maximum_tolerated_idleness
        #             residual = min(residual, self.TARGET_VIOLATION_FACTOR)  # 10
        #             rew += - residual
        #
        # rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        # rew = -100 if s.vector()[a] == 0 else rew

        # n_steps = int(self.simulator.max_travel_time() / config.DELTA_DEC)
        # rew = min_max_normalizer(rew,
        #                          startLB=(-(self.TARGET_VIOLATION_FACTOR * self.N_ACTIONS)) * n_steps - self.simulator.penalty_on_bs_expiration,
        #                          startUB=(self.N_ACTIONS)*n_steps,
        #                          endLB=-1,
        #                          endUB=1,
        #                          active=False)
        # return rew

    def evaluate_reward(self, s, a, s_prime, drone):
        return self.__rew_on_target(s, a, s_prime) if config.IS_DECIDED_ON_TARGET else self.__rew_on_flight(s, a, s_prime, drone)

    def evaluate_is_final_state(self, s, a, s_prime):
        """ The residual of the base station is >= 1, i.e. it is expired. """
        return s_prime.aoi_idleness_ratio(False)[0] >= 1  # or s.position() == s_prime.position()

    def invoke_train(self, drone):
        if drone.previous_state is None or drone.previous_action is None:
            return 0, self.previous_epsilon, self.previous_loss, False, None, None

        self.DQN.n_decision_step += 1

        s = drone.previous_state
        a = drone.previous_action
        s_prime = self.evaluate_state(drone)
        s_prime.is_final = self.evaluate_is_final_state(s, a, s_prime)
        r = self.evaluate_reward(s, a, s_prime, drone)

        if not drone.is_flying():
            s_prime._aoi_idleness_ratio[a] = 0  # set to 0 the residual of the just visited target (it will be reset later from drone.py)

        drone.previous_learning_tuple = s.vector(False, True), a, s_prime.vector(False, True), r

        if self.simulator.log_state >= 0:
            self.log_transition(s, s_prime, a, r, every=self.simulator.log_state, drone=drone)

        # print(s.position(False), a, s_prime.position(False), r)
        self.previous_epsilon = self.DQN.decay()

        IS_TRAIN = drone.identifier == self.simulator.n_drones - 1 # ony the last drone actually trains the NN
        # Continuous Tasks: Reinforcement Learning tasks which are not made of episodes, but rather last forever.
        # This tasks have no terminal states. For simplicity, they are usually assumed to be made of one never-ending episode.
        self.previous_loss = self.DQN.train(previous_state=s.vector(),
                                            current_state=s_prime.vector(),
                                            action=a,
                                            reward=r,
                                            is_final=s_prime.is_final,
                                            do=IS_TRAIN)

        if s_prime.is_final:
            self.simulator.environment.reset_drones_targets(False)
            self.reset_MDP(drone)

        return r, self.previous_epsilon, self.DQN.current_loss, s_prime.is_final, s, s_prime

    def invoke_predict(self, state, drone):
        if state is None:
            state = self.evaluate_state(drone)

        action_index, q = self.DQN.predict(state.vector())

        if drone.is_flying() and drone.previous_action is not None:
            action_index = drone.previous_action

        drone.previous_state = state
        drone.previous_action = action_index
        self.environment.write_previous_actions_drones[drone.identifier] = action_index

        # set the lock for the other not to pick this action
        self.simulator.environment.targets[action_index].lock = drone
        return action_index, q[0]

    def reset_MDP(self, drone):
        drone.previous_state = None
        drone.previous_action = None

    def log_transition(self, s, s_prime, a, r, every=1, drone=None):
        print("From drone n", drone.identifier)
        # print(s.vector(True, True))
        # print(s_prime.vector(True, True))

        print(s_prime.aoi_idleness_ratio(False))
        print(s_prime.time_distances(False))
        print(s_prime.closests(False))
        print(s_prime.actions_past(True))

        print(a, r)
        print("---")

        if drone.identifier == self.simulator.n_drones-1:
            time.sleep(every)
