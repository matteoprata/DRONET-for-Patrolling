
from src.patrolling.metrics import Metrics
from src.patrolling.patrolling_DQN import PatrollingDQN
from src.utilities.utilities import euclidean_distance
from src.utilities import config
import numpy as np


class State:
    def __init__(self, aois, time_distances, position, aoi_norm, time_norm, position_norm):
        self._aois: list = aois
        self._time_distances: list = time_distances
        self._position = position

        self.aoi_norm = aoi_norm
        self.time_norm = time_norm
        self.position_norm = position_norm

    def aois(self, normalized=True):
        return self._aois if not normalized else self.normalize_feature(self._aois, self.aoi_norm)

    def time_distances(self, normalized=True):
        return self._time_distances if not normalized else self.normalize_feature(self._time_distances, self.time_norm)

    def position(self, normalized=True):
        return self._position if not normalized else self.normalize_feature(self._position, self.position_norm)

    @staticmethod
    def normalize_feature(feature, normal_factor):
        return np.array(feature) * 1 / normal_factor

    @staticmethod
    def round_feature_vector(feature, rounding_digit):
        return [round(i, rounding_digit) for i in feature]

    def normalized_vector(self):
        return [self.position()] + list(self.aois()) + list(self.time_distances())

    def __repr__(self):
        return "{}\n{}\n{}".format(self.round_feature_vector(self._aois, 3), self.round_feature_vector(self._time_distances, 3), self._position)


class RLModule:
    def __init__(self, drone):
        self.drone = drone
        self.previous_state = None
        self.previous_action = None
        self.com_rewards = 0

        self.metrics = Metrics(simulator=drone.simulator)

        self.DQN = PatrollingDQN(pretrained_model_path=config.RL_MODEL,
                                 n_actions=self.drone.simulator.n_targets + self.drone.simulator.n_base_stations,
                                 n_features=2*(self.drone.simulator.n_targets + self.drone.simulator.n_base_stations)+1,
                                 simulator=self.drone.simulator,
                                 metrics=self.metrics
                                 )

        self.AOI_NORM = self.drone.simulator.sim_duration_ts * self.drone.simulator.ts_duration_sec
        self.TIME_NORM = self.drone.simulator.max_travel_time()
        self.ACTION_NORM = (self.drone.simulator.n_targets + self.drone.simulator.n_drones)

    def get_current_AOIs(self):
        return [target.relative_aoi() for target in self.drone.simulator.environment.targets]

    def get_current_time_distances(self):
        return [euclidean_distance(self.drone.coords, target.coords)/self.drone.speed for target in self.drone.simulator.environment.targets]

    def evaluate_state(self):
        pa = self.previous_action if self.previous_action is not None else 0

        AOI_NORM = self.drone.simulator.sim_duration_ts * self.drone.simulator.ts_duration_sec
        TIME_NORM = self.drone.simulator.max_travel_time()
        ACTION_NORM = (self.drone.simulator.n_targets + self.drone.simulator.n_drones)

        return State(self.get_current_AOIs(), self.get_current_time_distances(), pa, AOI_NORM, TIME_NORM, ACTION_NORM)

    def evaluate_reward(self, state):
        return sum(state.aois())

    def invoke_train(self):
        if self.previous_state is None or self.previous_action is None:
            return

        s = self.previous_state
        a = self.previous_action
        s_prime = self.evaluate_state()
        r = self.evaluate_reward(s_prime)

        self.metrics.cum_reward += r
        self.metrics.cum_aois += np.average(s_prime.aois(False))

        # Continuous Tasks: Reinforcement Learning tasks which are not made of episodes, but rather last forever.
        # This tasks have no terminal states. For simplicity, they are usually assumed to be made of one never-ending episode.
        self.DQN.train(previous_state=s.normalized_vector(), current_state=s_prime.normalized_vector(), action=a, reward=r)

    def invoke_predict(self):
        s = self.evaluate_state()
        action_index = self.DQN.predict(s.normalized_vector())

        self.previous_state = s
        self.previous_action = action_index
        return action_index

    def plot_statistics(self):
        FINE_PLOTS, PLOT = 50, 100

        self.metrics.losses.append(self.metrics.cum_loss)
        self.metrics.rewards.append(self.metrics.cum_reward)
        self.metrics.aois.append(self.metrics.cum_aois)
        self.metrics.epsilon.append(self.DQN.decay(self.DQN.n_decision_step, self.DQN.epsilon_decay))
        self.metrics.reset_counters()

        if self.DQN.n_decision_step > 0 and self.DQN.n_decision_step % FINE_PLOTS == 0:
            print("simulated step {}, decision step {}, current epsilon {}".format(self.drone.simulator.cur_step,
                                                                                   self.DQN.n_decision_step,
                                                                                   self.DQN.decay(self.DQN.n_decision_step, self.DQN.epsilon_decay)))

        if self.DQN.n_decision_step > 0 and self.DQN.n_decision_step % PLOT == 0:
            self.DQN.save_model()
            self.metrics.plot(self.drone.simulator.cur_step)

