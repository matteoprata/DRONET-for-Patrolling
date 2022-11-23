
from src.patrolling.patrolling_DQN_TORCH import PatrollingDQN
from src.patrolling.StateMDP import State

from src.evaluation.MetricsEvaluation import MetricsEvaluation
from src.utilities.utilities import euclidean_distance, min_max_normalizer
from src.utilities import config
import numpy as np
import time
from src.utilities.constants import JSONFields
from scipy.special import softmax


class RLModule:
    def __init__(self, drone):
        self.drone = drone
        self.simulator = self.drone.simulator

        self.previous_state = None
        self.previous_action = None
        self.previous_epsilon = 1
        self.previous_loss = None
        self.previous_learning_tuple = None
        
        self.N_ACTIONS = len(self.simulator.environment.targets)
        self.N_FEATURES = 2 * len(self.simulator.environment.targets)
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

        self.cumulative_reward = 0
        self.cumulative_loss = 0

        self.last_epoch_id = 0    # to check if we are in a new epoch // episode
        self.last_episode_id = 0

        self.learning_tuple = None

        self.validation_stats = np.zeros(shape=(config.TIME_DENSITY_METRICS,
                                                self.simulator.n_targets,
                                                self.simulator.n_episodes_validation))

    def __get_current_aoi_idleness_ratio(self, next=0):
        res = []
        for target in self.simulator.environment.targets:
            # set target need to 0 if this target is not necessary or is locked (OR)
            # -- target is locked from another drone (not this drone)
            # -- is inactive
            # TODO CHECK IGNORE CONDITION
            is_ignore_target = (target.lock is not None and target.lock != self.drone) or not target.active
            res_val = 0 if is_ignore_target else min(target.AOI_ratio(next), self.TARGET_VIOLATION_FACTOR)
            res.append(res_val)
        return res

    def __get_current_time_distances(self):
        """ TIME of TRANSIT """
        return [euclidean_distance(self.drone.coords, target.coords)/self.drone.speed for target in self.drone.simulator.environment.targets]

    def __evaluate_state(self):
        # pa = self.previous_action if self.previous_action is not None else 0
        # is_flying = self.drone.is_flying()
        # objective = self.previous_action if self.drone.is_flying() else self.N_ACTIONS + 1

        # - - # - - # - - # - - # - - # - - # - - # - - # - - #
        distances = self.__get_current_time_distances()
        residuals = self.__get_current_aoi_idleness_ratio()

        state = State(residuals, distances, None, self.AOI_NORM, self.TIME_NORM, None, False, None, None)
        return state

    def __rew_on_flight(self, s, a, s_prime):
        # if config.IS_RESIDUAL_REWARD:
        #     rew = self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        #     pos = 0
        #     neg = rew
        #     for tar_i in s_prime.aoi_idleness_ratio(False):
        #         val = max(1-tar_i, -self.TARGET_VIOLATION_FACTOR)
        #         if val < 0:
        #             neg += val
        #         else:
        #             pos += val
        #
        #     # normalize negative rewards
        #     rew = min_max_normalizer(neg,
        #                              startLB=(-(self.TARGET_VIOLATION_FACTOR * self.N_ACTIONS)
        #                                       + self.simulator.penalty_on_bs_expiration),
        #                              startUB=0,
        #                              endLB=-1,
        #                              endUB=0)
        #
        #     # normalize positive rewards then sum
        #     rew += min_max_normalizer(pos,
        #                              startUB=self.N_ACTIONS,
        #                              startLB=0,
        #                              endLB=0,
        #                              endUB=1/self.TARGET_VIOLATION_FACTOR)  # my guess
        #
        # else:
        #     sum_exp_res = - sum([min(i, self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False) if i >= 1])
        #     rew = sum_exp_res + (self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0)
        #     rew = min_max_normalizer(rew,
        #                              startLB=(- (self.TARGET_VIOLATION_FACTOR * self.N_ACTIONS) + self.simulator.penalty_on_bs_expiration),
        #                              startUB=0,
        #                              endLB=-1,
        #                              endUB=0

        # # REWARD TEST ATP01
        # rew = - max([min(i, self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False)])
        # rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        #
        # rew = min_max_normalizer(rew,
        #                          startUB=0,
        #                          startLB=(-(self.TARGET_VIOLATION_FACTOR - self.simulator.penalty_on_bs_expiration)),
        #                          endUB=0,
        #                          endLB=-1)

        # # REWARD TEST ATP02
        # rew = - sum([min(i, self.TARGET_VIOLATION_FACTOR) for i in s_prime.aoi_idleness_ratio(False)])
        # rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        #
        # rew = min_max_normalizer(rew,
        #                          startUB=0,
        #                          startLB=(-(self.TARGET_VIOLATION_FACTOR * self.N_ACTIONS - self.simulator.penalty_on_bs_expiration)),
        #                          endUB=0,
        #                          endLB=-1)

        rew = 0
        if not self.drone.is_flying():
            age = s.aoi_idleness_ratio(normalized=True)[a]  # if 0 means it looped
            rew = (1/age) / self.TIME_NORM if age > 0 else 0
            # print("Visited! Got reward", rew, "{}".format(age))
        return rew

    def __rew_on_target(self, s, a, s_prime):
        rew = 0

        time_dist_to_a = s.time_distances(False)[a]
        n_steps = max(int(time_dist_to_a/config.DELTA_DEC), 1)

        for step in range(n_steps):
            TIME = self.simulator.current_second() - (config.DELTA_DEC * step)

            for target in self.simulator.environment.targets:
                LAST_VISIT = target.last_visit_ts * self.simulator.ts_duration_sec
                if config.IS_RESIDUAL_REWARD:
                    pass  # check again
                    # residual = 1 - (TIME - LAST_VISIT) / target.maximum_tolerated_idleness
                    # residual = max(residual, -self.TARGET_VIOLATION_FACTOR) if residual >= 1 else 0 # 10
                    # rew += residual
                else:
                    residual = (TIME - LAST_VISIT) / target.maximum_tolerated_idleness
                    residual = min(residual, self.TARGET_VIOLATION_FACTOR) if residual >= 1 else 0  # 10
                    rew += - residual

        rew += self.simulator.penalty_on_bs_expiration if s_prime.is_final else 0
        rew = min_max_normalizer(rew,
                                 startLB=(-(self.TARGET_VIOLATION_FACTOR * self.N_ACTIONS))
                                         * n_steps - self.simulator.penalty_on_bs_expiration,
                                 startUB=0,
                                 endLB=-1,
                                 endUB=0)
        return rew

    def __evaluate_reward(self, s, a, s_prime):
        return self.__rew_on_target(s, a, s_prime) if config.IS_DECIDED_ON_TARGET else self.__rew_on_flight(s, a, s_prime)

    def __evaluate_is_final_state(self, s, a, s_prime):
        """ The residual of the base station is >= 1, i.e. it is expired. """
        return s_prime.aoi_idleness_ratio(False)[0] >= 1  # or s.position() == s_prime.position()

    def invoke_train(self):
        if self.previous_state is None or self.previous_action is None:
            return 0, self.previous_epsilon, self.previous_loss, False, None, None

        self.DQN.n_decision_step += 1

        s = self.previous_state
        a = self.previous_action
        s_prime = self.__evaluate_state()
        s_prime.is_final = self.__evaluate_is_final_state(s, a, s_prime)
        r = self.__evaluate_reward(s, a, s_prime)

        if not self.drone.is_flying():
            s_prime._aoi_idleness_ratio[a] = 0  # set to 0 the residual of the just visited target (it will be reset later from drone.py)

        self.previous_learning_tuple = s.vector(False, True), a, s_prime.vector(False, True), r  # for visualization

        # if self.simulator.log_state >= 0:
        #     self.log_transition(s, s_prime, a, r, every=self.simulator.log_state)

        # print(s.position(False), a, s_prime.position(False), r)
        self.previous_epsilon = self.DQN.let_exploration_decay()

        # Continuous Tasks: Reinforcement Learning tasks which are not made of episodes, but rather last forever.
        # These tasks have no terminal states. For simplicity, they are usually assumed to be made of one never-ending episode.
        self.previous_loss = self.DQN.train(previous_state=s.vector(),
                                            current_state=s_prime.vector(),
                                            action=a,
                                            reward=r,
                                            is_final=s_prime.is_final)

        if s_prime.is_final:
            self.simulator.reset_episode()

        return r, self.previous_epsilon, self.DQN.current_loss, s_prime.is_final, s, s_prime

    def invoke_predict(self, state):
        if state is None:
            state = self.__evaluate_state()

        action_index, q = self.DQN.predict(state.vector())

        # if self.drone.is_flying() and self.previous_action is not None:
        #     action_index = self.previous_action

        self.previous_state = state
        self.previous_action = action_index

        # set the lock for the other not to pick this action
        # self.simulator.environment.targets[action_index].lock = self.drone
        return action_index, q[0]

    def reset_MDP_episode(self):
        self.previous_state = None
        self.previous_action = None

    def log_transition(self, s, s_prime, a, r, every=1):
        print(s.vector(False, True))
        print(s_prime.vector(False, True))
        print(a, r)
        print("---")
        time.sleep(every)

    def log_training_metrics(self, logger):

        if self.simulator.is_validation:
            if self.is_new_episode():
                met = self.simulator.previous_metricsV2
                for t in range(1, self.simulator.n_targets):
                    _, y = MetricsEvaluation.AOI_func(t,
                                                      met.to_store_dictionary[JSONFields.VISIT_TIMES.value],
                                                      self.simulator.n_drones,
                                                      self.simulator.episode_duration,
                                                      self.simulator.ts_duration_sec,
                                                      met.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE.value],
                                                      density=config.TIME_DENSITY_METRICS)
                    
                    self.validation_stats[:, t, self.simulator.i_episode - 1] = y
            if self.is_last_validation_episode():
                print("Last episode")
                print(self.validation_stats.shape)
                # compute metrics
                # reset matrix
        return

        if logger is not None:
            if self.simulator.is_validation:
                if self.is_new_episode():
                    met = self.simulator.previous_metricsV2
                    print(met.to_store_dictionary)
                    exit()
                    for t in range(1, self.simulator.n_targets):
                        _, y = MetricsEvaluation.AOI_func(t,
                                                          met.to_store_dictionary[JSONFields.VISIT_TIMES.value],
                                                          self.simulator.n_drones,
                                                          self.simulator.episode_duration,
                                                          self.simulator.ts_duration_sec,
                                                          met.to_store_dictionary[JSONFields.SIMULATION_INFO.value][JSONFields.TOLERANCE.value])
                        self.validation_stats[:, t, self.simulator.i_episode-1] = y
                if self.is_last_validation_episode():
                    print("Last episode")
                    print(self.validation_stats.shape)
                    # compute metrics
                    # reset matrix
                    pass
            else:
                reward, epsilon, loss, _, _, _ = self.learning_tuple

                if self.is_new_epoch():

                    metrics = {"cumulative_reward": self.cumulative_reward,
                               "experience": epsilon,
                               "loss": self.cumulative_loss}

                    # add stats to track at the end of an epoch
                    logger.log(metrics)

                    self.cumulative_reward = 0
                    self.cumulative_loss = 0

                self.cumulative_reward += reward
                self.cumulative_loss += loss

    def is_new_epoch(self):
        if self.simulator.i_epoch == 0:
            self.last_epoch_id = 0

        if self.simulator.i_epoch != self.last_epoch_id:
            self.last_epoch_id = self.simulator.i_epoch
            return True
        return False

    def is_new_episode(self):
        if self.simulator.i_episode == 0:
            self.last_episode_id = 0

        if self.simulator.i_episode != self.last_episode_id:
            self.last_episode_id = self.simulator.i_episode
            return True
        return False

    def is_last_validation_episode(self):
        return self.simulator.i_episode == self.simulator.n_episodes_validation-1

    def is_decision_step(self):
        """ Whether is time to make a decision step or not """
        return (self.simulator.cur_step * self.simulator.ts_duration_sec) % config.DELTA_DEC == 0

    def query_policy(self):
        reward, epsilon, loss, is_end, s, s_prime = self.invoke_train()
        action, q = (0, None) if is_end else self.invoke_predict(s_prime)

        self.learning_tuple = reward, epsilon, loss, is_end, s, q
        self.log_training_metrics(self.simulator.wandb)

        return self.simulator.environment.targets[action]
