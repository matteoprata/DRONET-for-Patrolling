
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
            return list(self.aoi_idleness_ratio(normalized)) + list(self.time_distances(normalized)) #+ list(self.closests(normalized)) + list(self.actions_past(normalized))
        else:
            return [round(i, 2) for i in list(self.aoi_idleness_ratio(normalized))] + \
                   [round(i, 2) for i in list(self.time_distances(normalized))] #+ \
                   # [round(i, 2) for i in list(self.closests(normalized))] + \
                   # [round(i, 2) for i in list(self.actions_past(normalized))]

    def __repr__(self):
        str_state = "res: {}\ndis: {}\n" #clo: {}\nact: {}\n"
        return str_state.format(self.aoi_idleness_ratio(), self.time_distances()) #, self.closests(), self.actions_past()) #self.is_flying(False), self.objective(False))

    @staticmethod
    def round_feature_vector(feature, rounding_digit):
        return [round(i, rounding_digit) for i in feature]


class RLModule:
    def __init__(self, environment):
        self.environment = environment
        self.simulator = self.environment.simulator

        self.min_distances = []
        self.time_eval = None

        self.previous_epsilon = 1
        self.previous_loss = None

        self.is_final_episode_for_some = False

        self.N_ACTIONS = len(self.environment.targets)
        self.N_FEATURES = len(self.environment.targets) * 2 #+ len(self.environment.drones)
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
            res_val = 0 if is_ignore_target else min(target.aoi_idleness_ratio(next, drone.identifier), self.TARGET_VIOLATION_FACTOR)
            res_val = max(0, res_val)
            res.append(res_val)
        return res

    def get_current_time_distances(self, drone):
        """ TIME of TRANSIT """
        return [euclidean_distance(drone.coords, target.coords)/drone.speed for target in drone.simulator.environment.targets]

    def get_targets_closest_drone(self, drone):
        """ The shortest distance drone <> target for every target.  """

        if self.simulator.cur_step_total != self.time_eval:
            self.min_distances = []
            for tar in self.environment.targets:
                clo_time = np.inf
                for dr in self.environment.drones:
                    dis = euclidean_distance(dr.coords, tar.coords)
                    tem = dis / dr.speed
                    clo_time = tem if tem < clo_time else clo_time
                self.min_distances.append(clo_time)

        self.time_eval = self.simulator.cur_step_total

        return self.min_distances

    def prev_actions(self):
        return [0] * len(self.environment.drones) if self.environment.read_previous_actions_drones[0] is None else self.environment.read_previous_actions_drones

    def evaluate_state(self, drone):
        # - - # - - # - - # - - # - - # - - # - - # - - # - - #
        distances = self.get_current_time_distances(drone)      # N
        residuals = self.get_current_aoi_idleness_ratio(drone)  # N
        closests = None #self.get_targets_closest_drone(drone)        # N
        actions_past = None #self.prev_actions()                      # U

        state = State(residuals, distances, None, self.AOI_NORM, self.TIME_NORM, self.N_ACTIONS, False, None, None, closests, actions_past)
        return state

    def evaluate_reward(self, s, a, s_prime, drone):

        # # REWARD TEST ATP01
        # rew = - max([min(i, self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False)])
        # rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        # rew = min_max_normalizer(rew,
        #                          startUB=0,
        #                          startLB=(-(self.TARGET_VIOLATION_FACTOR - self.simulator.penalty_on_bs_expiration)),
        #                          endUB=0,
        #                          endLB=-1)

        # # REWARD TEST ATP02
        IS_EXPIRED_TARGET_CONDITION = config.IS_EXPIRED_TARGET_CONDITION
        rew = - sum([min(i, self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False) if i >= 1 or not IS_EXPIRED_TARGET_CONDITION])
        rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0

        rew = min_max_normalizer(rew,
                                 startUB=0,
                                 startLB=(-(self.TARGET_VIOLATION_FACTOR * self.N_ACTIONS - self.simulator.penalty_on_bs_expiration)),
                                 endUB=0,
                                 endLB=-1)

        return rew

    def evaluate_is_final_state(self, s, a, s_prime, drone):
        """ WARNING THIS IS TRUE FOR ALL DRONES AFTER THE """
        # SETS THIS TO TRUE UNTIL ALL DRONES ARE FINISHED
        self.is_final_episode_for_some = s_prime.aoi_idleness_ratio(False)[0] >= 1 or self.is_final_episode_for_some
        # self.final_episode_for_drone = drone if self.is_final_episode_for_some else self.final_episode_for_drone

        return s_prime.aoi_idleness_ratio(False)[0] >= 1  # or s.position() == s_prime.position()

    def invoke_train(self, drone):
        if drone.previous_state is None or drone.previous_action is None:
            return 0, self.previous_epsilon, self.previous_loss, False, None, None

        s = drone.previous_state
        a = drone.previous_action
        s_prime = self.evaluate_state(drone)
        s_prime.is_final = self.evaluate_is_final_state(s, a, s_prime, drone)
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

        # if s_prime.is_final:
        #     self.simulator.environment.reset_drones_targets(False)
        #     self.reset_MDP(drone)

        return r, self.previous_epsilon, self.DQN.current_loss, s_prime.is_final, s, s_prime

    def invoke_predict(self, state, drone):
        assert(len(self.environment.drones) <= len(self.environment.targets))
        state_attempt = state
        if state is None:
            state_attempt = self.evaluate_state(drone)

        # to avoid all of them heading to the same target
        if state is None and self.simulator.learning["is_pretrained"]:
            action_index = drone.identifier
        else:
            action_index = self.DQN.predict(state_attempt.vector())

        state = state_attempt
        if drone.is_flying() and drone.previous_action is not None:
            action_index = drone.previous_action

        drone.previous_state = state
        drone.previous_action = action_index
        self.environment.write_previous_actions_drones[drone.identifier] = action_index

        # set the lock for the other not to pick this action
        self.simulator.environment.targets[action_index].lock = drone
        return action_index

    def reset_MDP(self, drone):
        for d in self.environment.drones:
            d.previous_state = None
            d.previous_action = None

    def log_transition(self, s, s_prime, a, r, every=1, drone=None):
        print("From drone n", drone.identifier)

        print(s_prime)
        # print(s_prime.aoi_idleness_ratio(False))

        print(a, r)
        print("---")

        if drone.identifier == self.simulator.n_drones-1:
            time.sleep(every)
