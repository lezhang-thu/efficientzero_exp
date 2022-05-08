import gc
import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.utils import prepare_observation_lst, LinearSchedule


@ray.remote
class BatchWorker_CPU(object):
    def __init__(self, worker_id, replay_buffer, storage, mcts_storage,
                 config):
        """CPU Batch Worker for reanalyzing targets, see Appendix.
        Prepare the context concerning CPU overhead
        Parameters
        ----------
        worker_id: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        mcts_storage: Ant
            The mcts-related contexts storage
        """
        self.worker_id = worker_id
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.mcts_storage = mcts_storage
        self.config = config

        self.batch_max_num = 20
        self.beta_schedule = LinearSchedule(
            config.training_steps + config.last_steps,
            initial_p=config.priority_prob_beta,
            final_p=1.0)

    def make_batch(self, batch_context):
        """prepare the context of a batch
        reward_value_context:        the context of reanalyzed value targets
        policy_non_re_context:       the context of non-reanalyzed policy targets
        inputs_batch:                the inputs of batch
        Parameters
        ----------
        batch_context: Any
            batch context from replay buffer
        """
        # obtain the batch context from replay buffer
        game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_size = len(indices_lst)
        obs_lst, action_lst, mask_lst = [], [], []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_lst[i]
            game_pos = game_pos_lst[i]

            _actions = game.actions[game_pos:game_pos +
                                    self.config.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory)
            _mask = [1. for i in range(len(_actions))]
            _mask += [
                0. for _ in range(self.config.num_unroll_steps - len(_mask))
            ]

            _actions += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self.config.num_unroll_steps - len(_actions))
            ]

            # obtain the input observations
            obs_lst.append(game_lst[i].obs(
                game_pos_lst[i],
                extra_len=self.config.num_unroll_steps,
                padding=True))
            action_lst.append(_actions)
            mask_lst.append(_mask)

        # formalize the input observations
        obs_lst = prepare_observation_lst(obs_lst)

        # formalize the inputs of a batch
        inputs_batch = [
            obs_lst, action_lst, mask_lst, indices_lst, weights_lst,
            make_time_lst
        ]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        total_transitions = ray.get(self.replay_buffer.get_total_len.remote())

        # target reward, value
        batch_value_prefixs, batch_values = self._prepare_reward_value(
            indices_lst, game_lst, game_pos_lst, total_transitions)
        # target policy
        batch_policies = self._prepare_policy_non_re(indices_lst, game_lst,
                                                     game_pos_lst)
        targets_batch = [batch_value_prefixs, batch_values, batch_policies]
        self.mcts_storage.push([inputs_batch, targets_batch])

    def _prepare_policy_non_re(self, indices, games, state_index_lst):
        """just return the policy in self-play
        Parameters
        ----------
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        """
        batch_policies_non_re = []

        for game, state_index in zip(games, state_index_lst):
            traj_len = len(game)
            child_visit = game.child_visits

            target_policies = []
            for current_index in range(
                    state_index,
                    state_index + self.config.num_unroll_steps + 1):
                if current_index < traj_len:
                    target_policies.append(child_visit[current_index])
                else:
                    target_policies.append(
                        [0 for _ in range(self.config.action_space_size)])

            batch_policies_non_re.append(target_policies)

        batch_policies_non_re = np.asarray(batch_policies_non_re)
        return batch_policies_non_re

    def _prepare_reward_value(self, indices, games, state_index_lst,
                              total_transitions):
        """prepare the context of rewards and values for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        total_transitions: int
            number of collected transitions
        """
        config = self.config
        device = config.device
        batch_values, batch_value_prefixs = [], []

        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            reward_lst = game.rewards
            root_values = game.root_values

            target_values = []
            target_value_prefixs = []

            value_prefix = 0.0
            horizon_id = 0

            # off-policy correction: shorter horizon of td steps
            delta_td = (total_transitions - idx) // config.auto_td_steps
            td_steps = config.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, 5).astype(np.int)

            for current_index in range(
                    state_index, state_index + config.num_unroll_steps + 1):
                bootstrap_index = current_index + td_steps
                if bootstrap_index < traj_len:
                    value = root_values[bootstrap_index]
                else:
                    value = 0
                value *= config.discount**td_steps
                for i, reward in enumerate(
                        reward_lst[current_index:bootstrap_index]):
                    value += reward * config.discount**i

                # reset every lstm_horizon_len
                if horizon_id % config.lstm_horizon_len == 0:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    target_values.append(value)
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    value_prefix += reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_values.append(0)
                    target_value_prefixs.append(value_prefix)

            batch_value_prefixs.append(target_value_prefixs)
            batch_values.append(target_values)

        return np.asarray(batch_value_prefixs), np.asarray(batch_values)

    def run(self):
        # start making mcts contexts to feed the GPU batch maker
        start = False
        while True:
            # debug - start
            gc.collect()
            # debug - end

            # wait for starting
            if not start:
                start = ray.get(self.storage.get_start_signal.remote())
                time.sleep(1)
                continue

            trained_steps = ray.get(self.storage.get_counter.remote())
            beta = self.beta_schedule.value(trained_steps)
            # obtain the batch context from replay buffer
            batch_context = ray.get(
                self.replay_buffer.prepare_batch_context.remote(
                    self.config.batch_size, beta))
            # break
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(30)
                break
            if self.mcts_storage.get_len() < 20:
                self.make_batch(batch_context)
