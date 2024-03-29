
import numpy as np
from src.utilities import utilities as util

import torch
from src.config import Configuration

from src.RL.DQN import DQN
import src.constants as cst
from src.constants import LearningHyperParameters
from collections import namedtuple, deque
import random


class PatrollingDQN:
    def __init__(self, cf: Configuration, simulator, n_actions, n_state_features):

        self.cf = cf
        self.sim = simulator
        self.dqn_par = self.cf.DQN_PARAMETERS

        self.n_actions = n_actions
        self.n_state_features = n_state_features

        # build neural models

        # MODEL 1
        self.model = DQN(
            n_state_features,
            n_actions,
            [self.dqn_par[LearningHyperParameters.N_HIDDEN_1],
             self.dqn_par[LearningHyperParameters.N_HIDDEN_2],
             self.dqn_par[LearningHyperParameters.N_HIDDEN_3],
             self.dqn_par[LearningHyperParameters.N_HIDDEN_4],
             self.dqn_par[LearningHyperParameters.N_HIDDEN_5]]
        ).to(cst.TORCH_DEVICE)

        # MODEL 2
        self.model_hat = DQN(
            n_state_features,
            n_actions,
            [self.dqn_par[LearningHyperParameters.N_HIDDEN_1],
             self.dqn_par[LearningHyperParameters.N_HIDDEN_2],
             self.dqn_par[LearningHyperParameters.N_HIDDEN_3],
             self.dqn_par[LearningHyperParameters.N_HIDDEN_4],
             self.dqn_par[LearningHyperParameters.N_HIDDEN_5]]
        ).to(cst.TORCH_DEVICE)

        self.model_hat.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.dqn_par[LearningHyperParameters.LEARNING_RATE], amsgrad=True)

        if self.cf.is_rl_testing():
            print("Loading pretrained model at path", self.cf.RL_BEST_MODEL_PATH)
            self.model = torch.load(self.cf.RL_BEST_MODEL_PATH)
            self.model_hat = torch.load(self.cf.RL_BEST_MODEL_PATH)

        self.memory = ReplayMemory(self.dqn_par[LearningHyperParameters.REPLAY_MEMORY_DEPTH], self.sim.rnd_sample_replay)
        self.n_training_step = 0
        self.explore_prob = 1

    def is_explore_probability(self):
        """ Returns True if it is time to explore, False if it is time to exploit. """

        def explore_probability(step, exp_coeff, base=np.e):
            return base ** (-step * exp_coeff)

        def let_exploration_decay():
            """ Probability of exploration now. """
            self.explore_prob = explore_probability(self.sim.epoch, self.dqn_par[LearningHyperParameters.EPSILON_DECAY])
            return self.explore_prob

        return util.flip_biased_coin(p=let_exploration_decay(), random_gen=self.sim.rnd_explore)

    def predict(self, state, is_allowed_explore=True):
        """  Given an input state_prime, it returns the action predicted by the model if no exploration is done
          and the model is given as an input, if the exploration goes through. """

        is_do_explore = self.is_explore_probability()
        if is_do_explore and is_allowed_explore:
            # EXPLORE
            action_index = self.sim.rnd_explore.randint(0, self.n_actions)
            if state[action_index] == 0 and not self.cf.IS_ALLOW_SELF_LOOP:  # loop
                actions_available = list(range(self.n_actions))
                actions_available.pop(action_index)
                action_sub_index = self.sim.rnd_explore.randint(0, self.n_actions-1)
                action_index = actions_available[action_sub_index]
        else:
            # EXPLOIT
            state = torch.tensor(np.asarray(state).astype(np.float32)).to(cst.TORCH_DEVICE)
            with torch.no_grad():
                q_values = self.model(state)

            q_values = q_values.cpu()
            action_index = np.argmax(q_values)
            action_index = int(action_index)  # ?

            if state[action_index] == 0 and not self.cf.IS_ALLOW_SELF_LOOP:  # loop
                q_values[action_index] = - np.inf

            action_index = np.argmax(q_values)  # second best
            action_index = int(action_index)

        return action_index

    def train(self, previous_state, current_state, action, reward, is_NON_final):
        """ train the NN accumulate the experience and each X data the method actually train the network. """

        if None not in [previous_state, current_state, action, reward]:
            experience = (previous_state, current_state, action, reward, is_NON_final)
            self.memory.push(*experience)

        if self.time_to_batch_training():
            # print("Train", self.n_epochs, self.n_training_step)
            # sample at random from replay memory, batch_size elements
            random_sample_batch = self.memory.sample(self.dqn_par[LearningHyperParameters.BATCH_SIZE])
            self.__train_model_batched(random_sample_batch)

    def __train_model_batched(self, random_sample_batch):
        """ Given an input batch, it trains the model. """
        self.n_training_step += 1
        batch = Transition(*zip(*random_sample_batch))

        previous_states_v = torch.tensor(np.asarray(batch.previous_states).astype(np.float32)).to(cst.TORCH_DEVICE)
        current_states_v = torch.tensor(np.asarray(batch.current_states).astype(np.float32)).to(cst.TORCH_DEVICE)
        is_non_finals_mask = torch.BoolTensor(np.asarray(batch.is_NON_final)).to(cst.TORCH_DEVICE)

        non_final_current_states_v = current_states_v[is_non_finals_mask]

        actions_v = torch.tensor(np.asarray(batch.actions).astype(np.int64))  .to(cst.TORCH_DEVICE)
        rewards_v = torch.tensor(np.asarray(batch.rewards).astype(np.float32)).to(cst.TORCH_DEVICE)

        # Q-VALUES for all the actions of the batch
        actions_r = actions_v.reshape(len(actions_v), 1)  # BATCH x 1
        old_out = self.model(previous_states_v).gather(1, actions_r)
        old_out = old_out.reshape(1, len(old_out))[0]

        current_states_values = torch.zeros(self.cf.DQN_PARAMETERS[LearningHyperParameters.BATCH_SIZE]).to(cst.TORCH_DEVICE)
        with torch.no_grad():
            current_states_values[is_non_finals_mask] = self.model_hat(non_final_current_states_v).max(1)[0]

        expected_q = rewards_v + self.dqn_par[LearningHyperParameters.DISCOUNT_FACTOR] * current_states_values
        loss = torch.nn.SmoothL1Loss()(old_out, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        # print(self.decay(), "steps", self.sim.cur_step, "/", self.sim.sim_duration_ts)
        # print('swapped', self.n_training_step)
        self.swap_learning_model()

        current_loss = loss.item()

        self.sim.epoch_loss += [current_loss]
        self.sim.epoch_cumrew += [np.sum(rewards_v.cpu().numpy())]
        self.epoch_model = self.model

    def swap_learning_model(self):
        """ Updates the knowledge of the two """

        if self.time_to_swap_models():
            model_param = self.model.state_dict()
            model_hat_param = self.model_hat.state_dict()

            for key in model_param:
                tau = self.dqn_par[LearningHyperParameters.PERCENTAGE_SWAP]
                model_hat_param[key] = model_param[key] * tau + model_hat_param[key] * (1 - tau)

            self.model_hat.load_state_dict(model_hat_param)

    def save_model(self, path):
        torch.save(self.model, path)

    def time_to_batch_training(self, k=1):
        # and self.n_decison_step % self.batch_training_every_decision == 0
        return len(self.memory) > k * self.dqn_par[LearningHyperParameters.BATCH_SIZE]

    def time_to_swap_models(self):
        return self.n_training_step % self.dqn_par[LearningHyperParameters.SWAP_MODELS_EVERY_DECISION] == 0


Transition = namedtuple('Transition', ("previous_states", "current_states", "actions", "rewards", "is_NON_final"))


class ReplayMemory:

    def __init__(self, capacity, random_state):
        self.memory = deque([], maxlen=capacity)
        self.random_state = random_state

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """ Sampling is not sequential."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
