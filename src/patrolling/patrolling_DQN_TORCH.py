
import numpy as np
from src.utilities import config, utilities as util

import torch
from pytorch_lightning import LightningModule


class PatrollingDQN:
    def __init__(self,
                 pretrained_model_path,
                 n_actions,
                 n_features,
                 n_hidden_neurons_lv1,
                 n_hidden_neurons_lv2,
                 n_hidden_neurons_lv3,
                 simulator,
                 metrics,
                 batch_size=32,
                 lr=0.0001,
                 discount_factor=.99,
                 replay_memory_depth=100000,
                 swap_models_every_decision=500,
                 is_load_model=True,
                 ):

        self.simulator = simulator
        self.metrics = metrics
        self.batch_size = batch_size
        self.device = "cpu"
        self.lr = lr

        # number of actions, actions, number of states
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden_neurons_lv1 = n_hidden_neurons_lv1
        self.n_hidden_neurons_lv2 = n_hidden_neurons_lv2
        self.n_hidden_neurons_lv3 = n_hidden_neurons_lv3
        self.n_decision_step = 0

        # learning parameters
        self.discount_factor = discount_factor
        self.epsilon_decay = self.compute_epsilon_decay()
        self.replay_memory = util.LimitedList(replay_memory_depth)
        self.swap_models_every_decision = swap_models_every_decision

        # make the simulation reproducible
        np.random.seed(self.simulator.sim_seed)
        # tf.set_random_seed(self.simulator.sim_seed)
        self.is_load_model = is_load_model
        self.current_loss = None

        # build neural models
        if not self.is_load_model:
            self.model = DQN(self.n_features,
                             self.n_hidden_neurons_lv1,
                             self.n_hidden_neurons_lv2,
                             self.n_hidden_neurons_lv3,
                             self.n_actions)      # MODEL 1

            self.model_hat = DQN(self.n_features,
                             self.n_hidden_neurons_lv1,
                             self.n_hidden_neurons_lv2,
                             self.n_hidden_neurons_lv3,
                             self.n_actions)      # MODEL 2

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.model = torch.load(pretrained_model_path)
            self.model_hat = torch.load(pretrained_model_path)

    def compute_epsilon_decay(self, zero_perc_simulation=config.EXPLORE_PORTION, prob_threshold=config.ZERO_TOLERANCE):
        # keep the experience > .0001 until the first %80 of the steps
        # e^(- step_with_zero_exp * epsilon_decay) = 10^-4 -> - step_with_zero_exp * epsilon_decay = log_e 10^-4
        sim_duration = self.simulator.episode_duration * self.simulator.n_episodes * self.simulator.n_epochs
        step_with_zero_exp = sim_duration * zero_perc_simulation
        return - np.log(prob_threshold) / step_with_zero_exp

    @staticmethod
    def explore_probability(step, exp_coeff, base=np.e):
        return base ** (-step*exp_coeff)

    def let_exploration_decay(self):
        """ Probability of exploration now. """
        explore_prob = self.explore_probability(self.simulator.cur_step_total, self.epsilon_decay)
        return explore_prob

    def flip_biased_coin(self, p):
        """ Return true with probability p, false with probability 1-p. """
        return self.simulator.rnd_explore.random() < p

    def is_explore_probability(self):
        """ Returns True if it is time to explore, False otherwise. """
        return self.flip_biased_coin(self.let_exploration_decay())

    @torch.no_grad()
    def predict(self, state, is_explore=True):
        """  Given an input state, it returns the action predicted by the model if no exploration is done
          and the model is given as an input, if the exploration goes through. """

        if self.is_load_model:
            is_explore = False

        state = np.asarray(state).astype(np.float32)
        state = torch.tensor(state).to(self.device)
        q_values = self.model(state)

        if is_explore and self.is_explore_probability():
            action_index = self.simulator.rnd_explore.randint(0, self.n_actions)
            if state[action_index] == 0 and not config.IS_ALLOW_SELF_LOOP:  # loop
                actions_available = list(range(self.n_actions))
                actions_available.pop(action_index)
                action_sub_index = self.simulator.rnd_explore.randint(0, self.n_actions-1)
                action_index = actions_available[action_sub_index]
        else:
            action_index = np.argmax(q_values)
            action_index = int(action_index)

            if state[action_index] == 0 and not config.IS_ALLOW_SELF_LOOP:  # loop
                q_values[action_index] = - np.inf

            action_index = np.argmax(q_values)
            action_index = int(action_index)

        return action_index, q_values

    def train(self, previous_state=None, current_state=None, action=None, reward=None, is_final=None):
        """ train the NN accumulate the experience and each X data the method actually train the network. """
        if self.is_load_model:
            return

        if not (previous_state is None and current_state is None and action is None and reward is None and is_final is None):
            experience = [previous_state, current_state, action, reward, is_final]
            self.replay_memory.append(experience)

        if self.time_to_batch_training():
            # print("Train", self.n_epochs, self.n_decision_step)
            # sample at random from replay memory, batch_size elements
            random_sample_batch_indices = self.simulator.rstate_sample_batch_training.randint(0, len(self.replay_memory), size=self.batch_size)
            random_sample_batch = [self.replay_memory.llist[i] for i in random_sample_batch_indices]
            random_sample_batch = list(zip(*random_sample_batch))  # STATES, STATES, ACTIONS...

            return self.__train_model_batched(random_sample_batch)

    def __train_model_batched(self, random_sample_batch):
        """ Given an input batch, it trains the model. """

        previous_states, current_states, actions, rewards, is_finals = random_sample_batch

        previous_states_v = torch.tensor(np.asarray(previous_states).astype(np.float32)).to(self.device)
        current_states_v = torch.tensor(np.asarray(current_states)  .astype(np.float32)).to(self.device)
        actions_v = torch.tensor(np.asarray(actions)                .astype(np.int64)).to(self.device)
        rewards_v = torch.tensor(np.asarray(rewards)                .astype(np.float32)).to(self.device)
        is_finals_mask = torch.BoolTensor(is_finals).to(self.device)

        # Q-VALUES for all the actions of the batch
        old_out = self.model(previous_states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            cur_out = self.model_hat(current_states_v).max(1)[0]
            cur_out[is_finals_mask] = 0.0
            cur_out = cur_out.detach()

        expected_q = rewards_v + self.discount_factor * cur_out
        loss = torch.nn.MSELoss()(old_out, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.time_to_swap_models():
            # print(self.decay(), "steps", self.simulator.cur_step, "/", self.simulator.sim_duration_ts)
            # print('swapped', self.n_decision_step)
            self.swap_learning_model()

        self.current_loss = loss.item()
        return self.current_loss

    def swap_learning_model(self):
        """ Updates the knowledge of the two """
        # del self.model_hat
        # self.model_hat = self.build_neural_net()
        # self.model_hat.set_weights(self.model.get_weights())
        self.model_hat.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model, path)

    def time_to_batch_training(self):
        return len(self.replay_memory) > self.batch_size  # and self.n_decison_step % self.batch_training_every_decision == 0

    def time_to_swap_models(self):
        return self.n_decision_step % self.swap_models_every_decision == 0


class DQN(LightningModule):
    def __init__(self, in_shape, hidden_shape1, hidden_shape2, hidden_shape3, out_shape):
        super(DQN, self).__init__()

        if hidden_shape2 == 0 and hidden_shape3 == 0:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(in_shape, hidden_shape1),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_shape1, out_shape),
            )

        elif hidden_shape2 != 0 and hidden_shape3 == 0:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(in_shape, hidden_shape1),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_shape1, hidden_shape2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_shape2, out_shape),
            )

        elif hidden_shape2 != 0 and hidden_shape3 != 0:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(in_shape, hidden_shape1),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_shape1, hidden_shape2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_shape2, hidden_shape3),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_shape3, out_shape),
            )
        else:
            print("NN Layers setup is unexpected.")
            exit()

    def forward(self, x):
        return self.fc(x)

