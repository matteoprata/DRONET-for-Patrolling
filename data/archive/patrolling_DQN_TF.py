import random

import numpy as np
from src.utilities import utilities as util
from src import config

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class PatrollingDQN:
    def __init__(self,
                 pretrained_model_path,
                 n_actions,
                 n_features,
                 simulator,
                 metrics,
                 batch_size=32,
                 lr=0.0001,
                 discount_factor=.99,  # .99,
                 replay_memory_depth=100000,
                 swap_models_every_decision=500,  # 10000
                 load_model=True,
                 ):
        self.simulator = simulator
        self.metrics = metrics
        self.batch_size = batch_size

        self.model_loss = 0
        self.avg_reward = 0
        self.beta = beta
        self.lr = lr

        # number of actions, actions, number of states
        self.n_actions = n_actions
        self.n_features = n_features

        self.n_decision_step = 0
        self.n_epochs = 1

        # learning parameters
        self.discount_factor = discount_factor
        self.epsilon_decay = self.compute_epsilon_decay()
        self.replay_memory = util.LimitedList(replay_memory_depth)
        self.swap_models_every_decision = swap_models_every_decision

        # make the simulation reproducible
        # tf.set_random_seed(self.sim.sim_seed)

        # sess = tf.Session(graph=tf.get_default_graph())
        # tf.keras.backend.set_session(sess)

        self.load_model = load_model
        self.current_loss = None

        # build neural models
        if not self.load_model:
            self.model = self.build_neural_net()      # MODEL 1
            self.model_hat = self.build_neural_net()  # MODEL 2
            print(self.model.summary())
        else:
            # import os
            # assert os.visited_targets_coordinates.isdir("./model")
            # exit()
            self.model = keras.models.load_model(pretrained_model_path)
            self.model_hat = keras.models.load_model(pretrained_model_path)

    def compute_epsilon_decay(self, zero_perc_simulation=config.EXPLORE_PORTION, prob_threshold=config.ZERO_TOLERANCE):
        # keep the experience > .0001 until the first %80 of the steps
        # e^(- step_with_zero_exp * epsilon_decay) = 10^-4 -> - step_with_zero_exp * epsilon_decay = log_e 10^-4
        sim_duration = self.simulator.episode_duration * self.simulator.n_episodes * self.simulator.n_epochs
        step_with_zero_exp = sim_duration * zero_perc_simulation
        return - np.log(prob_threshold) / step_with_zero_exp

    @staticmethod
    def explore_probability(step, exp_coeff, base=np.e):
        return base ** (-step*exp_coeff)

    def decay(self):
        """ Probability of exploration now. """
        explore_prob = self.explore_probability(self.simulator.cur_step_total, self.epsilon_decay)
        return explore_prob

    def flip_biased_coin(self, p):
        """ Return true with probability p, false with probability 1-p. """
        return self.simulator.rnd_explore.random() < p

    def is_explore_probability(self):
        """ Returns True if it is time to explore, False otherwise. """
        return self.flip_biased_coin(self.decay())

    def build_neural_net(self):
        """ Construct the model from scratch """

        # -------------- Q-NN module body --------------
        model = Sequential()

        n_hidden_neurons = int(np.sqrt(self.n_features * self.n_actions))
        model.add(Dense(n_hidden_neurons, input_dim=self.n_features, activation='relu'))
        model.add(Dense(self.n_actions))

        opt = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(loss='mse', optimizer=opt)
        # model.compile(optimizer="adam", loss="mean_squared_error", optimizer=opt)

        return model

    def predict(self, state, is_explore=True):
        """  Given an input state_prime, it returns the action predicted by the model if no exploration is done
          and the model is given as an input, if the exploration goes through. """

        if self.load_model:
            is_explore = False

        q_values = self.model.predict(np.array([state]))
        if is_explore and self.is_explore_probability():
            action_index = self.simulator.rnd_explore.randint(0, self.n_actions)
            # q_values = np.asarray([0]*self.n_actions)  # crafted for visualization
            # q_values[action_index] = np.inf
            # print("random", action_index)
        else:
            # q_values = self.model.predict(np.array([state_prime]))  # q-values for the input state_prime
            action_index = np.argmax(q_values[0])
            # print("q", action_index)
        return action_index, q_values

    def train(self, previous_state=None, current_state=None, action=None, reward=None, is_final=None):
        """ train the NN accumulate the experience and each X data the method actually train the network. """
        if self.load_model:
            return

        if is_final:
            self.n_epochs += 1

        if not (previous_state is None and current_state is None and action is None and reward is None and is_final is None):
            experience = [previous_state, current_state, action, reward, is_final]
            self.replay_memory.append(experience)

        if self.time_to_batch_training():
            # print("Train", self.n_epochs, self.n_training_step)
            # sample at random from replay memory, batch_size elements
            random_sample_batch_indices = self.simulator.rstate_sample_batch_training.randint(0, len(self.replay_memory), size=self.batch_size)
            random_sample_batch = [self.replay_memory.llist[i] for i in random_sample_batch_indices]

            return self.__train_model_batched(random_sample_batch)

    def __train_model_batched(self, random_sample_batch):
        """ Given an input batch, it trains the model. """

        X, y = [], []  # batches input - output (correct prediction)
        for previous_state, current_state, action, reward, is_final in random_sample_batch:
            old_out = self.model.predict(np.array([previous_state]))
            cur_out = self.model_hat.predict(np.array([current_state]))

            old_out[0, action] = (reward - self.avg_reward) + self.discount_factor * np.max(cur_out[0]) if not is_final else reward
            # self.avg_reward += 0  # self.beta * old_out[0, action]
            X.append(previous_state)
            y.append(old_out[0])

        training_result = self.model.fit(np.array(X), np.array(y), epochs=1, batch_size=self.batch_size, verbose=0)
        self.current_loss = training_result.history["loss"][0]

        if self.time_to_swap_models():
            # print(self.decay(), "steps", self.sim.cur_step, "/", self.sim.sim_duration_ts)
            # print('swapped', self.n_training_step)
            self.swap_learning_model()

        return self.current_loss

    def swap_learning_model(self):
        """ Updates the knowledge of the two """
        # del self.model_hat
        # self.model_hat = self.build_neural_net()
        self.model_hat.set_weights(self.model.get_weights())

    def save_model(self, path):
        self.model.save(path)

    def time_to_batch_training(self):
        return len(self.replay_memory) > self.batch_size  # and self.n_decison_step % self.batch_training_every_decision == 0

    def time_to_swap_models(self):
        return self.n_decision_step % self.swap_models_every_decision == 0

