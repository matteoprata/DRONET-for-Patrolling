
import numpy as np
from src.utilities import utilities as util
from src import config

import torch
from src.config import Configuration, LearningHyperParameters
from src.constants import PatrollingProtocol

from src.patrolling.DQN_NN2 import DQN
from src.patrolling.ReplayMemory2 import ReplayMemory, Transition
import src.constants as cst


class PatrollingDQN:
    def __init__(self, cf: Configuration, simulator, n_actions, n_state_features):

        self.cf = cf
        self.sim = simulator
        self.dqn_par = self.cf.DQN_PARAMETERS

        self.n_actions = n_actions
        self.n_state_features = n_state_features

        # build neural models
        if self.cf.is_rl_training():

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
            print(self.dqn_par[LearningHyperParameters.LEARNING_RATE])
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.dqn_par[LearningHyperParameters.LEARNING_RATE], amsgrad=True)

        elif self.cf.is_rl_testing():

            self.model = torch.load(self.cf.RL_BEST_MODEL_PATH)
            self.model_hat = torch.load(self.cf.RL_BEST_MODEL_PATH)

        self.memory = ReplayMemory(self.dqn_par[LearningHyperParameters.REPLAY_MEMORY_DEPTH], self.sim.rnd_sample_replay)
        self.n_training_step = 0

    def is_explore_probability(self):
        """ Returns True if it is time to explore, False if it is time to exploit. """

        def explore_probability(step, exp_coeff, base=np.e):
            return base ** (-step * exp_coeff)

        def let_exploration_decay():
            """ Probability of exploration now. """
            explore_prob = explore_probability(self.n_training_step, self.dqn_par[LearningHyperParameters.EPSILON_DECAY])
            return explore_prob

        return util.flip_biased_coin(let_exploration_decay(), random_gen=self.sim.rnd_explore)

    def predict(self, state, is_allowed_explore=True):
        """  Given an input state, it returns the action predicted by the model if no exploration is done
          and the model is given as an input, if the exploration goes through. """

        state = np.asarray(state).astype(np.float32)
        state = torch.tensor(state).to(cst.TORCH_DEVICE)

        with torch.no_grad():
            q_values = self.model(state)

        do_exp = self.is_explore_probability()

        if do_exp and is_allowed_explore:
            action_index = self.sim.rnd_explore.randint(0, self.n_actions)
            if state[action_index] == 0 and not self.cf.IS_ALLOW_SELF_LOOP:  # loop
                actions_available = list(range(self.n_actions))
                actions_available.pop(action_index)
                action_sub_index = self.sim.rnd_explore.randint(0, self.n_actions-1)
                action_index = actions_available[action_sub_index]
        else:
            q_values = q_values.cpu()
            action_index = np.argmax(q_values)
            action_index = int(action_index)  # ?

            if state[action_index] == 0 and not self.cf.IS_ALLOW_SELF_LOOP:  # loop
                q_values[action_index] = - np.inf

            action_index = np.argmax(q_values)  # second best
            action_index = int(action_index)

        return action_index

    def train(self, previous_state=None, current_state=None, action=None, reward=None, is_final=None):
        """ train the NN accumulate the experience and each X data the method actually train the network. """

        self.n_training_step += 1

        if None not in [previous_state, current_state, action, reward]:
            experience = (previous_state, current_state, action, reward)
            self.memory.push(*experience)

        if self.time_to_batch_training():
            # print("Train", self.n_epochs, self.n_training_step)
            # sample at random from replay memory, batch_size elements
            random_sample_batch = self.memory.sample(self.dqn_par[LearningHyperParameters.BATCH_SIZE])

            # random_sample_batch_indices = self.sim.rstate_sample_batch_training.randint(0, len(self.replay_memory), size=self.batch_size)
            # random_sample_batch = [self.replay_memory.llist[i] for i in random_sample_batch_indices]
            # random_sample_batch = list(zip(*random_sample_batch))  # STATES, STATES, ACTIONS...

            return self.__train_model_batched(random_sample_batch)

    def __train_model_batched(self, random_sample_batch):
        """ Given an input batch, it trains the model. """
        batch = Transition(*zip(*random_sample_batch))

        previous_states_v = torch.tensor(np.asarray(batch.previous_states).astype(np.float32)).to(cst.TORCH_DEVICE)
        current_states_v = torch.tensor(np.asarray(batch.current_states)  .astype(np.float32)).to(cst.TORCH_DEVICE)
        actions_v = torch.tensor(np.asarray(batch.actions)                .astype(np.int64))  .to(cst.TORCH_DEVICE)
        rewards_v = torch.tensor(np.asarray(batch.rewards)                .astype(np.float32)).to(cst.TORCH_DEVICE)
        # is_finals_mask = torch.BoolTensor(is_finals).to(cst.TORCH_DEVICE)

        # Q-VALUES for all the actions of the batch
        actions_r = actions_v.reshape(len(actions_v), 1)  # BATCH x 1
        old_out = self.model(previous_states_v).gather(1, actions_r)
        old_out = old_out.reshape(1, len(old_out))[0]

        with torch.no_grad():
            cur_out = self.model_hat(current_states_v).max(1)[0]
            # cur_out[is_finals_mask] = 0.0
            # cur_out = cur_out.detach()

        expected_q = rewards_v + self.dqn_par[LearningHyperParameters.DISCOUNT_FACTOR] * cur_out
        loss = torch.nn.SmoothL1Loss()(old_out, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        # print(self.decay(), "steps", self.sim.cur_step, "/", self.sim.sim_duration_ts)
        # print('swapped', self.n_training_step)
        self.swap_learning_model()

        self.current_loss = loss.item()

        self.sim.epoch_loss += [np.sum(rewards_v.cpu().numpy())]
        self.sim.epoch_cumrew += [self.current_loss]

        return self.current_loss

    def swap_learning_model(self, tau=0.005):
        """ Updates the knowledge of the two """

        if self.time_to_swap_models():
            model_param = self.model.state_dict()
            model_hat_param = self.model_hat.state_dict()

            for key in model_param:
                model_hat_param[key] = model_param[key] * tau + model_hat_param[key] * (1 - tau)

            self.model_hat.load_state_dict(model_hat_param)

    def save_model(self, path):
        torch.save(self.model, path)

    def time_to_batch_training(self, k=1):
        # and self.n_decison_step % self.batch_training_every_decision == 0
        return len(self.memory) > k * self.dqn_par[LearningHyperParameters.BATCH_SIZE]

    def time_to_swap_models(self):
        return self.n_training_step % self.dqn_par[LearningHyperParameters.SWAP_MODELS_EVERY_DECISION] == 0
