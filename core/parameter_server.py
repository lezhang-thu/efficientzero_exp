import ray
import time
import torch

import numpy as np
import torch.nn.functional as F


def consist_loss_func(f1, f2):
    """Consistency loss function: similarity loss
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


@ray.remote
class ParameterServer(object):
    def __init__(self, config):
        self.config = config
        self.model = config.get_uniform_network()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=config.lr_init,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

    def apply_gradients(self, *gradients, step_count=None):
        self._adjust_lr(step_count)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def _adjust_lr(self, step_count):
        config = self.config
        optimizer = self.optimizer
        # adjust learning rate, step lr every lr_decay_steps
        if step_count < config.lr_warm_step:
            lr = config.lr_init * step_count / config.lr_warm_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = config.lr_init * config.lr_decay_rate**(
                (step_count - config.lr_warm_step) // config.lr_decay_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


@ray.remote(num_gpus=0.2)
class GradActor:
    def __init__(self, config, replay_buffer, mcts_storage):
        self.config = config
        self.model = config.get_uniform_network().to(config.device)
        self.model.train()
        self.replay_buffer = replay_buffer
        self.mcts_storage = mcts_storage

    def _update_weights(self, batch):
        config = self.config
        inputs_batch, targets_batch = batch
        obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
        target_value_prefix, target_value, target_policy = targets_batch

        obs_batch_ori = torch.from_numpy(np.array(obs_batch_ori)).to(
            config.device).float() / 255.0
        obs_batch = obs_batch_ori[:, 0:config.stacked_observations *
                                  config.image_channel, :, :]
        obs_target_batch = obs_batch_ori[:, config.image_channel:, :, :]

        # do augmentations
        if config.use_augmentation:
            obs_batch = config.transform(obs_batch)
            obs_target_batch = config.transform(obs_target_batch)

        # use GPU tensor
        action_batch = torch.from_numpy(np.array(action_batch)).to(
            config.device).unsqueeze(-1).long()
        x = []
        for _ in [
                mask_batch, target_value_prefix, target_value, target_policy,
                weights_lst
        ]:
            x.append(torch.from_numpy(np.array(_)).to(config.device).float())
        mask_batch, target_value_prefix, target_value, target_policy, weights = x

        batch_size = obs_batch.size(0)
        assert batch_size == config.batch_size == target_value_prefix.size(0)

        # transform targets to categorical representation
        transformed_target_value_prefix = config.scalar_transform(
            target_value_prefix)
        target_value_prefix_phi = config.reward_phi(
            transformed_target_value_prefix)

        transformed_target_value = config.scalar_transform(target_value)
        target_value_phi = config.value_phi(transformed_target_value)

        value, _, policy_logits, hidden_state, reward_hidden = self.model.initial_inference(
            obs_batch)
        scaled_value = config.inverse_value_transform(value)
        predicted_value_prefixs = []

        # calculate the new priorities for each transition
        value_priority = torch.nn.L1Loss(reduction='none')(
            scaled_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy(
        ) + config.prioritized_replay_eps

        # loss of the first step
        value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
        policy_loss = -(torch.log_softmax(policy_logits, dim=1) *
                        target_policy[:, 0]).sum(1)
        value_prefix_loss = torch.zeros(batch_size, device=config.device)
        consistency_loss = torch.zeros(batch_size, device=config.device)

        gradient_scale = 1 / config.num_unroll_steps
        # loss of the unrolled steps
        for step_i in range(config.num_unroll_steps):
            # unroll with the dynamics function
            value, value_prefix, policy_logits, hidden_state, reward_hidden = self.model.recurrent_inference(
                hidden_state, reward_hidden, action_batch[:, step_i])

            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i +
                                                config.stacked_observations)

            # consistency loss
            if config.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                _, _, _, presentation_state, _ = self.model.initial_inference(
                    obs_target_batch[:, beg_index:end_index, :, :])
                # no grad for the presentation_state branch
                dynamic_proj = self.model.project(hidden_state, with_grad=True)
                observation_proj = self.model.project(presentation_state,
                                                      with_grad=False)
                temp_loss = consist_loss_func(
                    dynamic_proj, observation_proj) * mask_batch[:, step_i]
                consistency_loss += temp_loss

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) *
                             target_policy[:, step_i + 1]).sum(1)
            value_loss += config.scalar_value_loss(
                value, target_value_phi[:, step_i + 1])
            value_prefix_loss += config.scalar_reward_loss(
                value_prefix, target_value_prefix_phi[:, step_i])
            # Follow MuZero, set half gradient
            hidden_state.register_hook(lambda grad: grad * 0.5)

            # reset hidden states
            if (step_i + 1) % config.lstm_horizon_len == 0:
                reward_hidden = (torch.zeros(1, config.batch_size,
                                             config.lstm_hidden_size).to(
                                                 config.device),
                                 torch.zeros(1, config.batch_size,
                                             config.lstm_hidden_size).to(
                                                 config.device))

        # weighted loss with masks (some invalid states which are out of trajectory)
        loss = (config.consistency_coeff * consistency_loss +
                config.policy_loss_coeff * policy_loss +
                config.value_loss_coeff * value_loss +
                config.reward_loss_coeff * value_prefix_loss)
        weighted_loss = (weights * loss).mean()

        # backward
        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)

        self.model.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       config.max_grad_norm)

        # update priority
        new_priority = value_priority
        self.replay_buffer.update_priorities.remote(indices, new_priority,
                                                    make_time)
        return self.model.get_gradients()

    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        batch = None
        while True:
            # obtain a batch
            batch = self.mcts_storage.pop()
            if batch is not None:
                break
            else:
                time.sleep(1)
        return self._update_weights(batch)
