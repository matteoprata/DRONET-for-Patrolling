
import numpy as np
from src.utilities import config, utilities as util
import tensorflow.compat.v1 as tf

from keras.models import Sequential, Input
from keras.layers import Dense
from keras import optimizers
import keras as kr


class PatrollingDQN:
    def __init__(self,
                 pretrained_model_path,
                 n_actions,
                 n_features,
                 simulator,
                 metrics,
                 batch_size=32,
                 discount_factor=1,  # .99,
                 epsilon_decay=.0004,
                 replay_memory_depth=100000,
                 swap_models_every_decision=1000,  # 10000
                 load_model=False,
                 ):

        self.simulator = simulator
        self.metrics = metrics
        self.batch_size = batch_size

        self.model_loss = 0
        self.avg_reward = 0
        self.beta = 0.01

        # number of actions, actions, number of states
        self.n_actions = n_actions
        self.n_features = n_features

        self.n_decision_step = 0

        # learning parameters
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.replay_memory = util.LimitedList(replay_memory_depth)
        self.swap_models_every_decision = swap_models_every_decision

        # make the simulation reproducible
        np.random.seed(self.simulator.sim_seed)
        tf.set_random_seed(self.simulator.sim_seed)

        # sess = tf.Session(graph=tf.get_default_graph())
        # tf.keras.backend.set_session(sess)

        self.load_model = load_model

        # build neural models
        if not self.load_model:
            self.model = self.build_neural_net()      # MODEL 1
            self.model_hat = self.build_neural_net()  # MODEL 2
        else:
            self.model = kr.models.load_model(pretrained_model_path)
            self.model_hat = kr.models.load_model(pretrained_model_path)

    @staticmethod
    def decay(step, exp_coeff, base=np.e):
        return base ** (-step*exp_coeff)

    def flip_biased_coin(self, p):
        """ Return true with probability p, false with probability 1-p. """
        return self.simulator.rnd_explore.random() < p

    def is_explore_probability(self):
        """ Returns True if it is time to explore, False otherwise. """
        return self.flip_biased_coin(self.decay(self.n_decision_step, self.epsilon_decay))

    def build_neural_net(self):
        """ Construct the model from scratch """

        # -------------- Q-NN module body --------------

        model = Sequential()

        model.add(Dense(20, input_dim=self.n_features, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.n_actions))

        opt = optimizers.Adam(learning_rate=0.0001)  # (learning_rate=0.01)
        model.compile(loss='mean_squared_error', optimizer=opt)  # huber

        return model

    def predict(self, state, is_explore=True):
        """  Given an input state, it returns the action predicted by the model if no exploration is done
          and the model is given as an input, if the exploration goes through. """

        if self.load_model:
            is_explore = False

        if is_explore and self.is_explore_probability():
            action_index = self.simulator.rnd_explore.randint(0, self.n_actions)
        else:
            q_values = self.model.predict(np.asarray([state]))  # q-values for the input state
            action_index = np.argmax(q_values[0])
            # print(action_index, "was thaken from nn failed wp", self.decay(self.n_decison_step, self.epsilon_decay))

        # print(action_index, self.decay(self.n_training_step, self.epsilon_decay), "steps", self.simulator.cur_step, "/", self.simulator.sim_duration_ts)
        return action_index

    def train(self, previous_state=None, current_state=None, action=None, reward=None):
        """ train the NN accumulate the experience and each X data the method actually train the network. """
        if self.load_model:
            return

        if not (previous_state is None and current_state is None and action is None and reward is None):
            experience = [previous_state, current_state, action, reward]
            self.replay_memory.append(experience)

        if self.time_to_batch_training():
            # sample at random from replay memory, batch_size elements
            random_sample_batch_indices = self.simulator.rstate_sample_batch_training.randint(0, len(self.replay_memory), size=self.batch_size)
            random_sample_batch = np.asarray(self.replay_memory.llist)[random_sample_batch_indices]

            self.__train_model_batched(random_sample_batch)

    # def train_whole_memory(self, n_epochs):
    #     """ Trains for n_epoch the nn. """
    #
    #     print("starting final training over memory of size {}, training steps {}".format(len(self.replay_memory), self.n_training_step))
    #     for e in range(n_epochs):
    #         print("epoch", e)
    #         for b in tqdm(range(0, len(self.replay_memory), self.batch_size)):
    #             self.n_training_step += 1
    #             sample_batch = self.replay_memory.llist[b:b+self.batch_size]
    #             self.__train_model_batched(sample_batch)

    def __train_model_batched(self, random_sample_batch):
        """ Given an input batch, it trains the model. """

        X, y = [], []  # batches input - output (correct prediction)
        for previous_state, current_state, action, reward in random_sample_batch:
            old_out = self.model.predict(np.asarray([previous_state]))
            cur_out = self.model_hat.predict(np.asarray([current_state]))

            old_out[0, action] = (reward - self.avg_reward) + self.discount_factor * np.max(cur_out[0])
            self.avg_reward += 0 # self.beta * old_out[0, action]

            X.append(previous_state)
            y.append(old_out[0])

        training_result = self.model.fit(np.asarray(X), np.asarray(y), epochs=1, batch_size=self.batch_size, verbose=0)
        self.metrics.cum_loss += training_result.history["loss"][0]

        if self.time_to_swap_models():
            print(self.decay(self.n_decision_step, self.epsilon_decay), "steps", self.simulator.cur_step, "/", self.simulator.sim_duration_ts)
            print('swapped', self.n_decision_step)
            self.swap_learning_model()

    def swap_learning_model(self):
        """ Updates the knowledge of the two """
        # del self.model_hat
        # self.model_hat = self.build_neural_net()
        self.model_hat.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save(config.RL_DATA+self.simulator.simulation_name())

    def time_to_batch_training(self):
        return len(self.replay_memory) > self.batch_size  # and self.n_decison_step % self.batch_training_every_decision == 0

    def time_to_swap_models(self):
        return self.n_decision_step % self.swap_models_every_decision == 0
