import cProfile
import io
import json
import os
import pstats
import sys
import time
from json import JSONDecodeError
from typing import Iterator, List, Tuple, Union
from weakref import ref

import numba
import numpy as np
import torch
import torch as th
from torch.distributions import kl_divergence
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.policy import Policy
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils.util import (
    calculate_prob_last_selector_action_at_step_for_steps,
)


class PPO:
    """
    Proximal Policy Optimization algorithm (PPO)

    :param rollout_generator: Function that will generate the rollouts
    :param agent: An ActorCriticAgent
    :param n_steps: The number of steps to run per update
    :param gamma: Discount factor
    :param batch_size: batch size to break experience data into for training
    :param epochs: Number of epoch when optimizing the loss
    :param minibatch_size: size to break batch sets into (helps combat VRAM issues)
    :param clip_range: PPO Clipping parameter for the value function
    :param ent_coef: Entropy coefficient for the loss calculation
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: optional clip_grad_norm value
    :param logger: wandb logger to store run results
    :param device: torch device
    :param zero_grads_with_none: 0 gradient with None instead of 0

    Look here for info on zero_grads_with_none
    https://pytorch.org/docs/master/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad
    """

    def __init__(
            self,
            rollout_generator: BaseRolloutGenerator,
            agent: ActorCriticAgent,
            n_steps=4096,
            gamma=0.99,
            batch_size=512,
            epochs=10,
            # reuse=2,
            minibatch_size=None,
            clip_range=0.2,
            ent_coef=0.01,
            gae_lambda=0.95,
            vf_coef=1,
            max_grad_norm=0.5,
            logger=None,
            device="cuda",
            zero_grads_with_none=False,
            kl_models_weights: List[
                Union[Tuple[Policy, float], Tuple[Policy, float, float]]
            ] = None,
            disable_gradient_logging=False,
            reward_logging_dir=None,
            target_clip_frac=None,
            min_lr=1e-7,
            max_lr=1,
            clip_frac_kp=0.5,
            clip_frac_ki=0,
            clip_frac_kd=0,
            wandb_wait_btwn=5,
            save_latest=False,
            action_selection_dict=None,
            num_actions=0,
            extra_prints=False,
    ):
        self.is_selector = rollout_generator.selector_skip_k is not None
        if self.is_selector:
            assert (
                    kl_models_weights is None
            ), "Cannot use selector with a KL divergence loss term"

        self.extra_prints = extra_prints
        self.num_actions = num_actions
        self.action_selection_dict = action_selection_dict
        self.save_latest = save_latest
        self.rollout_generator = rollout_generator
        self.reward_logging_dir = reward_logging_dir
        # TODO let users choose their own agent
        # TODO move agent to rollout generator
        self.agent = agent.to(device)
        self.device = device
        self.zero_grads_with_none = zero_grads_with_none
        self.frozen_iterations = 0
        self._saved_lr = None

        self.wandb_wait_btwn = wandb_wait_btwn

        self.starting_iteration = 0

        # hyperparameters
        self.epochs = epochs
        self.gamma = gamma
        # assert n_steps % batch_size == 0
        # self.reuse = reuse
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        assert self.batch_size % self.minibatch_size == 0
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr_pid_cont = PIDLearningRateController(
            target_clip_frac, clip_frac_kp, clip_frac_ki, clip_frac_kd, min_lr, max_lr
        )
        self.target_clip_frac = target_clip_frac

        self.running_rew_mean = 0
        self.running_rew_var = 1
        self.running_rew_count = 1e-4

        self.total_steps = 0
        self.logger = logger
        if not disable_gradient_logging:
            self.logger.watch((self.agent.actor, self.agent.critic))
        self.timer = time.time_ns() // 1_000_000
        self.jit_tracer = None

        if kl_models_weights is not None:
            for i in range(len(kl_models_weights)):
                assert len(kl_models_weights[i]) in (2, 3)
                if len(kl_models_weights[i]) == 2:
                    kl_models_weights[i] = kl_models_weights[i] + (None,)
        self.kl_models_weights = kl_models_weights

        # clean up previous files
        if self.reward_logging_dir is not None:
            files = os.listdir(self.reward_logging_dir)
            for file in files:
                file = os.path.join(self.reward_logging_dir, file)
                os.unlink(file)

    def update_reward_norm(self, rewards: np.ndarray) -> np.ndarray:
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = rewards.shape[0]

        delta = batch_mean - self.running_rew_mean
        tot_count = self.running_rew_count + batch_count

        new_mean = self.running_rew_mean + delta * batch_count / tot_count
        m_a = self.running_rew_var * self.running_rew_count
        m_b = batch_var * batch_count
        m_2 = (
                m_a
                + m_b
                + np.square(delta)
                * self.running_rew_count
                * batch_count
                / (self.running_rew_count + batch_count)
        )
        new_var = m_2 / (self.running_rew_count + batch_count)

        new_count = batch_count + self.running_rew_count

        self.running_rew_mean = new_mean
        self.running_rew_var = new_var
        self.running_rew_count = new_count

        return (rewards - self.running_rew_mean) / np.sqrt(
            self.running_rew_var + 1e-8
        )  # TODO normalize before update?

    def run(
            self,
            iterations_per_save=10,
            save_dir=None,
            save_jit=False,
            end_after_steps=None,
    ):
        """
        Generate rollout data and train
        :param iterations_per_save: number of iterations between checkpoint saves
        :param save_dir: where to save
        """
        if save_dir:
            current_run_dir = os.path.join(
                save_dir, self.logger.project + "_" + str(time.time())
            )
            os.makedirs(current_run_dir)
        elif iterations_per_save and not save_dir:
            print("Warning: no save directory specified.")
            print("Checkpoints will not be saved.")

        iteration = self.starting_iteration
        rollout_gen = self.rollout_generator.generate_rollouts()

        self.rollout_generator.update_parameters(
            self.agent.actor, iteration, self.total_steps
        )  # noqa
        last_wandb_call = 0

        while True:
            # pr = cProfile.Profile()
            # pr.enable()
            t0 = time.time()

            def _iter():
                size = 0
                wasted_data = 0
                old_data = 0
                new_data = 0
                print(f"Collecting rollouts ({iteration})...")
                while size < self.n_steps:
                    try:
                        rollout = next(rollout_gen)
                        wasted_data += self.rollout_generator.wasted_data  # noqa
                        old_data += self.rollout_generator.old_data  # noqa
                        new_data += self.rollout_generator.new_data  # noqa
                        if rollout.learnable_size() > 0:
                            size += rollout.learnable_size()
                            # progress.update(rollout.size())
                            yield rollout
                    except StopIteration:
                        return
                perc_old_data = old_data / (old_data + new_data)
                self.logger.log(
                    {"ppo/%old_data": perc_old_data, "ppo/wasted_data": wasted_data},
                    commit=False,
                )
                print(f"%old data: {perc_old_data}  --- wasted data: {wasted_data} ")

            self.calculate(_iter(), iteration)
            iteration += 1

            if save_dir:
                if self.save_latest:
                    self.save(
                        os.path.join(save_dir, self.logger.project + "_" + "latest"),
                        -1,
                        save_jit,
                    )
                if iteration % iterations_per_save == 0:
                    self.save(current_run_dir, iteration, save_jit)  # noqa

            if self.frozen_iterations > 0:
                if self.frozen_iterations == 1:
                    print(" ** Unfreezing policy network **")

                    assert self._saved_lr is not None
                    self.agent.optimizer.param_groups[0]["lr"] = self._saved_lr
                    self._saved_lr = None

                self.frozen_iterations -= 1

            self.rollout_generator.update_parameters(
                self.agent.actor, iteration - 1, self.total_steps
            )  # noqa

            # calculate years for graph
            # if self.tick_skip_starts is not None:
            #     new_iteration = iteration
            #     years = 0
            #     for i in reversed(self.tick_skip_starts):
            #         length = new_iteration - i[1]
            #         years += length * i[2] / (3600 * 24 * 365 * (120 / i[0]))
            #         new_iteration = i[1]
            #     self.logger.log({"ppo/years": years}, step=iteration, commit=False)

            # add reward log outputs here with commit false
            if self.reward_logging_dir is not None:
                self.log_rewards(iteration - 1)

            self.total_steps += self.n_steps  # size
            t1 = time.time()
            commit = False
            if t1 - last_wandb_call > self.wandb_wait_btwn:
                commit = True
            self.logger.log(
                {
                    "ppo/steps_per_second": self.n_steps / (t1 - t0),
                    "ppo/total_timesteps": self.total_steps,
                },
                step=iteration - 1,
                commit=commit,
            )
            print(f"fps: {self.n_steps / (t1 - t0)}\ttotal steps: {self.total_steps}")

            # pr.disable()
            # s = io.StringIO()
            # sortby = pstats.SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.dump_stats(f"profile_{self.total_steps}")
            if end_after_steps is not None:
                if self.total_steps >= end_after_steps:
                    break

    def set_logger(self, logger):
        self.logger = logger

    def log_rewards(self, iteration):
        # need to read all of the json in the directory, handle them, delete them
        # adding the stats for sliders
        files = os.listdir(self.reward_logging_dir)
        num_files = 0
        num_steps = 0
        total_dict = {}
        avg_dict = {}
        total_dict_blue = {}
        total_dict_orange = {}
        avg_dict_blue = {}
        avg_dict_orange = {}
        abs_dict = {}
        # require minimum number of files
        if len(files) < 50:
            return
        for file in files:
            num_files += 1
            try:
                file = os.path.join(self.reward_logging_dir, file)
                fh = open(file)
                data = json.load(fh)
                num_steps += data.get("step_num")
                current_steps = data.get("step_num")
                num_players = data.get("num_players")
                mid = num_players // 2
                # just sum all of the sums and then divide by num_files later to get average episode total
                # divide by number of players
                for key, value in data.get("RewardSum").items():
                    if key in total_dict:
                        total_dict[key] += sum(value) / num_players
                        abs_dict[key] += sum(abs(v) for v in value) / num_players
                    else:
                        total_dict[key] = sum(value) / num_players
                        abs_dict[key] = sum(abs(v) for v in value) / num_players
                    if data.get("kickoff"):
                        if key in total_dict_blue:
                            total_dict_blue[key] += (
                                                            sum(value[:mid]) / mid
                                                    ) / num_players
                        else:
                            total_dict_blue[key] = (
                                                           sum(value[:mid]) / mid
                                                   ) / num_players
                        if key in total_dict_orange:
                            total_dict_orange[key] += (
                                                              sum(value[mid:]) / mid
                                                      ) / num_players
                        else:
                            total_dict_orange[key] = (
                                                             sum(value[mid:]) / mid
                                                     ) / num_players
                # to get average we need to weight the averages by steps
                w_1 = (
                        current_steps / num_steps
                )  # num_steps already includes current_steps
                w_2 = (num_steps - current_steps) / num_steps
                for key, value in data.get("RewardAvg").items():
                    if key in avg_dict:
                        avg_dict[key] = (
                                                avg_dict[key] * w_2 + sum(value) * w_1
                                        ) / num_players
                    else:
                        avg_dict[key] = sum(value) / num_players

                    # split into blue and orange
                    if data.get("kickoff"):
                        if key in avg_dict_blue:
                            avg_dict_blue[key] = (
                                    avg_dict_blue[key] * w_2
                                    + (sum(value[:mid]) / mid) * w_1
                            )
                        else:
                            avg_dict_blue[key] = sum(value[:mid]) / mid
                        if key in avg_dict_orange:
                            avg_dict_orange[key] = (
                                    avg_dict_orange[key] * w_2
                                    + (sum(value[mid:]) / mid) * w_1
                            )
                        else:
                            avg_dict_orange[key] = sum(value[mid:]) / mid

                fh.close()
                os.unlink(file)
            except JSONDecodeError:
                print(f"Error with json while working on file {file}")

        # divide the sum by number of episodes/aka files
        for key, value in total_dict.items():
            total_dict[key] = value / num_files
        for key, value in total_dict_blue.items():
            total_dict_blue[key] = value / num_files
        for key, value in total_dict_orange.items():
            total_dict_orange[key] = value / num_files
        for key, value in abs_dict.items():
            abs_dict[key] = value / num_files

        # thanks to WaddlestheTimePig for this idea
        # p = np.polyfit(slider_vals, ep_rew_avgs, 1)
        # fit = np.poly1d(p)(slider_vals)
        #
        # # Log these two values
        # fit_slope = p[0]
        # res_std = np.std(ep_rew_avgs - fit)

        # total_dict is the episode average, avg_dict is the per-step avg
        log_dict = {}
        # remove this per player version, it's messy and loud
        # log_dict.update(
        #     {f"rewards_ep_ind/{key}_{i}": val for key, values in total_dict.items() for i, val in enumerate(values)})
        # log_dict.update(
        #     {f"rewards_step_ind/{key}_{i}": val for key, values in avg_dict.items() for i, val in enumerate(values)})
        for key, value in total_dict.items():
            log_dict.update({f"rewards_ep/{key}": value})
        for key, value in avg_dict.items():
            log_dict.update({f"rewards_step/{key}": value})
        for key, value in total_dict_blue.items():
            log_dict.update({f"rewards_ep_team/{key}_blue": value})
        for key, value in total_dict_orange.items():
            log_dict.update({f"rewards_ep_team/{key}_orange": value})
        for key, value in avg_dict_blue.items():
            log_dict.update({f"rewards_step_team/{key}_blue": value})
        for key, value in avg_dict_orange.items():
            log_dict.update({f"rewards_step_team/{key}_orange": value})
        for key, value in abs_dict.items():
            log_dict.update({f"rewards_ep_abs/{key}": value})

        # sorted_dict = dict(sorted(log_dict.items()))  #  wandb doesn't respect this anyway
        self.logger.log(log_dict, step=iteration, commit=False)

    def evaluate_actions_selector(self, trajectory_observations, trajectory_actions):
        trajectory_batch_size = len(trajectory_observations[0]) if isinstance(trajectory_observations, tuple) else len(
            trajectory_observations)
        selector_choice_probs = calculate_prob_last_selector_action_at_step_for_steps(trajectory_batch_size - 1,
                                                                                      self.rollout_generator.selector_skip_probability_table)
        selector_choice_probs = th.as_tensor(selector_choice_probs, device='cuda')
        dist = self.agent.actor.get_action_distribution(trajectory_observations)
        dist_entropy = dist.entropy()
        log_prob_tensors = []
        entropy_tensors = []
        for step, action in enumerate(trajectory_actions):
            log_prob_tensors.append(th.log(th.sum(dist.probs[:, 0, action] * selector_choice_probs[step])).expand(1))
            entropy_tensors.append(th.sum(dist_entropy * selector_choice_probs[step]).expand(1))
        return th.cat(log_prob_tensors), th.cat(entropy_tensors)

    def evaluate_actions(self, observations, actions):
        """
        Calculate Log Probability and Entropy of actions
        """
        dist = self.agent.actor.get_action_distribution(observations)
        # indices = self.agent.get_action_indices(dists)

        log_prob = self.agent.actor.log_prob(dist, actions)
        entropy = self.agent.actor.entropy(dist, actions)

        if self.extra_prints:
            print(
                f"log prob mean: {log_prob.mean()}  min: {log_prob.min()}  max: {log_prob.max()}"
            )
            print(
                f"entropy mean: {entropy.mean()}  min: {entropy.min()}  max: {entropy.max()}"
            )
            print(f"dist (eval actions) prob min: {dist.probs.min()}")
            print(
                f"dist (eval actions) prob max: {dist.probs.max()}, prob mean: {dist.probs.mean()}"
            )

        entropy = -torch.mean(entropy)
        return log_prob, entropy, dist

    @staticmethod
    @numba.njit
    def _calculate_advantages_numba(rewards, values, gamma, gae_lambda, truncated):
        advantages = np.zeros_like(rewards)
        # v_targets = np.zeros_like(rewards)
        dones = np.zeros_like(rewards)
        # if truncated:
        #     print("got truncated in ppo")
        dones[-1] = 1.0 if not truncated else 0.0
        episode_starts = np.zeros_like(rewards)
        episode_starts[0] = 1.0
        last_values = values[-1]
        last_gae_lam = 0
        size = len(advantages)
        for step in range(size - 1, -1, -1):
            if step == size - 1:
                next_non_terminal = 1.0 - dones[-1].item()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1].item()
                next_values = values[step + 1]
            v_target = rewards[step] + gamma * next_values * next_non_terminal
            delta = v_target - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
            # v_targets[step] = v_target
        return advantages  # , v_targets

    def calculate(self, buffers: Iterator[ExperienceBuffer], iteration):
        """
        Calculate loss and update network
        """
        obs_tensors = []
        act_tensors = []
        # value_tensors = []
        log_prob_tensors = []
        # advantage_tensors = []
        returns_tensors = []

        rewards_tensors = []
        cur_policy_step_log_prob_tensors = []
        cur_policy_step_entropy_tensors = []

        ep_rewards = []
        ep_steps = []
        action_count = np.asarray([0] * self.num_actions)
        action_changes = 0
        num_unlearnable = 0

        n = 0

        for buffer in buffers:  # Do discounts for each ExperienceBuffer individually
            # this sections breaks on advanced obs, not sure if necessary at all?
            # if isinstance(buffer.observations[0], (tuple, list)):
            if isinstance(buffer.observations[0], list):
                transposed = tuple(zip(*buffer.observations))
                obs_tensor = tuple(
                    torch.from_numpy(np.vstack(t)).float() for t in transposed
                )
            else:  # use just this section for advanced obs
                obs_tensor = th.from_numpy(np.vstack(buffer.observations)).float()

            with th.no_grad():
                if isinstance(obs_tensor, tuple):
                    x = tuple(o.to(self.device) for o in obs_tensor)
                else:
                    x = obs_tensor.to(self.device)
                values = (
                    self.agent.critic(x).detach().cpu().numpy().flatten()
                )  # No batching?
                torch.cuda.empty_cache()  # adding to try to fix memory issues

            actions = np.stack(buffer.actions)
            log_probs = np.stack(buffer.log_probs)
            rewards = np.stack(buffer.rewards)
            dones = np.stack(buffer.dones)
            learnable_mask = np.stack(buffer.learnable)
            num_unlearnable += len(learnable_mask) - np.sum(learnable_mask)
            if self.is_selector:
                cur_policy_step_log_prob, cur_policy_step_entropy = (
                    self.evaluate_actions_selector(x, th.from_numpy(actions))
                )
                cur_policy_step_log_prob_tensors.append(cur_policy_step_log_prob)
                cur_policy_step_entropy_tensors.append(cur_policy_step_entropy)
            size = rewards.shape[0]

            advantages = self._calculate_advantages_numba(
                rewards, values, self.gamma, self.gae_lambda, dones[-1] == 2
            )

            returns = advantages + values
            if self.action_selection_dict is not None:
                flat_actions = actions[:, 0].flatten()
                unique, counts = np.unique(flat_actions, return_counts=True)
                for i, value in enumerate(unique):
                    action_count[value] += counts[i]
                action_changes += (np.diff(flat_actions) != 0).sum()
            if isinstance(obs_tensor, tuple):
                obs_tensors.append(
                    tuple(tensor[learnable_mask] for tensor in obs_tensor)
                )
            else:
                obs_tensors.append(obs_tensor[learnable_mask])
            act_tensors.append(th.from_numpy(actions[learnable_mask]))
            log_prob_tensors.append(th.from_numpy(log_probs[learnable_mask]))
            returns_tensors.append(th.from_numpy(returns[learnable_mask]))
            rewards_tensors.append(th.from_numpy(rewards[learnable_mask]))

            ep_rewards.append(rewards.sum())
            ep_steps.append(size)
            n += 1
        ep_rewards = np.array(ep_rewards)
        ep_steps = np.array(ep_steps)

        total_steps = sum(ep_steps)
        self.logger.log(
            {
                "ppo/ep_reward_mean": ep_rewards.mean(),
                "ppo/ep_reward_std": ep_rewards.std(),
                "ppo/ep_len_mean": ep_steps.mean(),
                "submodel_swaps/action_changes": action_changes / total_steps,
                "ppo/mean_reward_per_step": ep_rewards.mean() / ep_steps.mean(),
                "ppo/abs_ep_reward_mean": np.abs(ep_rewards).sum() / ep_steps.mean(),
                "ppo/unlearnable_count": num_unlearnable,
            },
            step=iteration,
            commit=False,
        )

        if self.action_selection_dict is not None:
            for k, v in self.action_selection_dict.items():
                count = action_count[k]
                name = "submodels/" + v
                ratio_used = count / total_steps
                self.logger.log({name: ratio_used}, step=iteration, commit=False)

        print(f"std, mean rewards: {ep_rewards.std()}\t{ep_rewards.mean()}")

        if isinstance(obs_tensors[0], tuple):
            transposed = zip(*obs_tensors)
            obs_tensor = tuple(th.cat(t).float() for t in transposed)
        else:
            obs_tensor = th.cat(obs_tensors).float()
        act_tensor = th.cat(act_tensors)
        log_prob_tensor = th.cat(log_prob_tensors).float()
        # advantages_tensor = th.cat(advantage_tensors)
        returns_tensor = th.cat(returns_tensors).float()
        cur_policy_step_log_prob_tensor = th.cat(cur_policy_step_log_prob_tensors)
        cur_policy_step_entropy_tensor = th.cat(cur_policy_step_entropy_tensors)

        tot_loss = 0
        tot_policy_loss = 0
        tot_entropy_loss = 0
        tot_value_loss = 0
        total_kl_div = 0
        tot_clipped = 0

        if self.kl_models_weights is not None:
            tot_kl_other_models = np.zeros(len(self.kl_models_weights))
            tot_kl_coeffs = np.zeros(len(self.kl_models_weights))

        n = 0

        if self.jit_tracer is None:
            self.jit_tracer = obs_tensor[0].to(self.device)

        print("Training network...")

        if self.frozen_iterations > 0:
            print("Policy network frozen, only updating value network...")

        precompute = torch.cat(
            [param.view(-1) for param in self.agent.actor.parameters()]
        )
        t0 = time.perf_counter_ns()
        self.agent.optimizer.zero_grad(set_to_none=self.zero_grads_with_none)
        for e in range(self.epochs):
            # this is mostly pulled from sb3
            indices = torch.randperm(returns_tensor.shape[0])[: self.batch_size]
            if isinstance(obs_tensor, tuple):
                obs_batch = tuple(o[indices] for o in obs_tensor)
                # obs_batch = tuple(obs_tensor[i][indices] for i in range(len(obs_tensor)))  # try to speed up
            else:
                obs_batch = obs_tensor[indices]
            act_batch = act_tensor[indices]
            log_prob_batch = log_prob_tensor[indices]
            # advantages_batch = advantages_tensor[indices]
            returns_batch = returns_tensor[indices]
            cur_policy_step_log_prob_batch = cur_policy_step_log_prob_tensor[indices]
            cur_policy_step_entropy_batch = cur_policy_step_entropy_tensor[indices]

            for i in range(0, self.batch_size, self.minibatch_size):
                # Note: Will cut off final few samples

                if isinstance(obs_tensor, tuple):
                    obs = tuple(
                        o[i: i + self.minibatch_size].to(self.device)
                        for o in obs_batch
                    )
                else:
                    obs = obs_batch[i: i + self.minibatch_size].to(self.device)

                act = act_batch[i: i + self.minibatch_size].to(self.device)
                # adv = advantages_batch[i:i + self.minibatch_size].to(self.device)
                ret = returns_batch[i: i + self.minibatch_size].to(self.device)
                old_log_prob = log_prob_batch[i: i + self.minibatch_size].to(
                    self.device
                )

                # TODO optimization: use forward_actor_critic instead of separate in case shared, also use GPU
                try:
                    if self.is_selector:
                        log_prob = cur_policy_step_log_prob_batch[i: i + self.minibatch_size]
                        entropy = cur_policy_step_entropy_batch[i: i + self.minibatch_size]
                        dist = None
                    else:
                        log_prob, entropy, dist = self.evaluate_actions(
                            obs, act
                        )  # Assuming obs and actions as input
                except ValueError as e:
                    print("ValueError in evaluate_actions", e)
                    continue
                diff_log_prob = log_prob - old_log_prob
                #  stabilize the ratio for small log prob
                ratio = torch.where(
                    diff_log_prob.abs() < 0.00005,
                    1 + diff_log_prob,
                    torch.exp(diff_log_prob),
                )
                if self.extra_prints:
                    print(
                        f"ratio.min is {ratio.min()}  max is {ratio.max()}  mean is {ratio.mean()}"
                    )
                values_pred = self.agent.critic(obs)

                values_pred = th.squeeze(values_pred)
                adv = ret - values_pred.detach()
                adv = (adv - th.mean(adv)) / (th.std(adv) + 1e-8)

                # clipped surrogate loss
                policy_loss_1 = adv * ratio
                low_side = 1 - self.clip_range
                policy_loss_2 = adv * th.clamp(ratio, low_side, 1 / low_side)
                if self.extra_prints:
                    print(
                        f"Policy_loss_1: mean: {policy_loss_1.mean()} min: {policy_loss_1.min()} max: {policy_loss_1.max()}"
                    )
                    print(
                        f"Policy_loss_2: mean: {policy_loss_2.mean()} min: {policy_loss_2.min()} max: {policy_loss_2.max()}"
                    )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # **If we want value clipping, add it here**
                value_loss = F.mse_loss(ret, values_pred)

                if entropy is None:
                    # Approximate entropy when no analytical form
                    print("Entropy is None, approximating")
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = entropy

                kl_loss = 0
                if self.kl_models_weights is not None:
                    for k, (model, kl_coef, half_life) in enumerate(
                            self.kl_models_weights
                    ):
                        if half_life is not None:
                            kl_coef *= 0.5 ** (self.total_steps / half_life)
                        with torch.no_grad():
                            dist_other = model.get_action_distribution(obs)
                        div = kl_divergence(dist_other, dist).mean()
                        tot_kl_other_models[k] += div
                        tot_kl_coeffs[k] = kl_coef
                        kl_loss += kl_coef * div

                loss = (
                               policy_loss
                               + self.ent_coef * entropy_loss
                               + self.vf_coef * value_loss
                               + kl_loss
                       ) / (self.batch_size / self.minibatch_size)

                if not torch.isfinite(loss).all():
                    print("Non-finite loss, skipping", n)
                    print("\tPolicy loss:", policy_loss)
                    print("\tEntropy loss:", entropy_loss)
                    print("\tValue loss:", value_loss)
                    print("\tTotal loss:", loss)
                    print("\tRatio:", ratio)
                    print("\tAdv:", adv)
                    print("\tLog prob:", log_prob)
                    print("\tOld log prob:", old_log_prob)
                    print("\tEntropy:", entropy)
                    print(
                        "\tActor has inf:",
                        any(
                            not p.isfinite().all()
                            for p in self.agent.actor.parameters()
                        ),
                    )
                    print(
                        "\tCritic has inf:",
                        any(
                            not p.isfinite().all()
                            for p in self.agent.critic.parameters()
                        ),
                    )
                    print("\tReward as inf:", not np.isfinite(ep_rewards).all())
                    if isinstance(obs, tuple):
                        for j in range(len(obs)):
                            print(f"\tObs[{j}] has inf:", not obs[j].isfinite().all())
                    else:
                        print("\tObs has inf:", not obs.isfinite().all())
                    continue

                loss.backward()

                # Unbiased low variance KL div estimator from http://joschu.net/blog/kl-approx.html
                total_kl_div += th.mean((ratio - 1) - (log_prob - old_log_prob)).item()
                tot_loss += loss.item()
                tot_policy_loss += policy_loss.item()
                tot_entropy_loss += entropy_loss.item()
                tot_value_loss += value_loss.item()
                tot_clipped += th.mean(
                    (th.abs(ratio - 1) > self.clip_range).float()
                ).item()
                n += 1
                # pb.update(self.minibatch_size)

            # Clip grad norm
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)

            self.agent.optimizer.step()
            self.agent.optimizer.zero_grad(set_to_none=self.zero_grads_with_none)

        # update the LR to keep it within the clip fraction
        if self.target_clip_frac is not None:
            if n == 0:
                print(
                    "no good epochs. The LR were: {} and {}",
                    self.agent.optimizer.param_groups[0]["lr"],
                    self.agent.optimizer.param_groups[1]["lr"],
                )
            else:
                orig_actor = self.agent.optimizer.param_groups[0]["lr"]
                self.agent.optimizer.param_groups[0]["lr"] = self.lr_pid_cont.adjust(
                    tot_clipped / n, self.agent.optimizer.param_groups[0]["lr"]
                )
                # self.agent.optimizer.param_groups[1]["lr"] = self.lr_pid_cont.adjust(tot_clipped / n, self.agent.optimizer.param_groups[1]["lr"])
                self.agent.optimizer.param_groups[1]["lr"] = (
                    self.agent.optimizer.param_groups[0]["lr"]
                )
                after_actor = self.agent.optimizer.param_groups[0]["lr"]
                print(
                    f"clipped {tot_clipped} and changed from {orig_actor} to {after_actor}"
                )

        t1 = time.perf_counter_ns()

        assert n > 0

        postcompute = torch.cat(
            [param.view(-1) for param in self.agent.actor.parameters()]
        )

        log_dict = {
            "ppo/loss": tot_loss / n,
            "ppo/policy_loss": tot_policy_loss / n,
            "ppo/entropy_loss": tot_entropy_loss / n,
            "ppo/value_loss": tot_value_loss / n,
            "ppo/mean_kl": total_kl_div / n,
            "ppo/clip_fraction": tot_clipped / n,
            "ppo/epoch_time": (t1 - t0) / (1e6 * self.epochs),
            "ppo/update_magnitude": th.dist(precompute, postcompute, p=2),
        }

        if self.target_clip_frac is not None:
            log_dict.update(
                {"ppo/actor_lr": self.agent.optimizer.param_groups[0]["lr"]}
            )
            log_dict.update(
                {"ppo/critic_lr": self.agent.optimizer.param_groups[1]["lr"]}
            )

        if self.kl_models_weights is not None and len(self.kl_models_weights) > 0:
            log_dict.update(
                {
                    f"ppo/kl_div_model_{i}": tot_kl_other_models[i] / n
                    for i in range(len(self.kl_models_weights))
                }
            )
            log_dict.update(
                {
                    f"ppo/kl_coeff_model_{i}": tot_kl_coeffs[i]
                    for i in range(len(self.kl_models_weights))
                }
            )

        self.logger.log(
            log_dict, step=iteration, commit=False
        )  # Is committed after when calculating fps

    def load(self, load_location, continue_iterations=True):
        """
        load the model weights, optimizer values, and metadata
        :param load_location: checkpoint folder to read
        :param continue_iterations: keep the same training steps
        """

        checkpoint = torch.load(load_location)
        self.agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if continue_iterations:
            self.starting_iteration = checkpoint["epoch"]
            self.total_steps = checkpoint["total_steps"]
            print("Continuing training at iteration " + str(self.starting_iteration))

    def save(self, save_location, current_step, save_actor_jit=False):
        """
        Save the model weights, optimizer values, and metadata
        :param save_location: where to save
        :param current_step: the current iteration when saved. Use to later continue training
        :param save_actor_jit: save the policy network as a torch jit file for rlbot use
        """

        version_str = str(self.logger.project) + "_" + str(current_step)
        version_dir = os.path.join(save_location, version_str)

        os.makedirs(version_dir, exist_ok=current_step == -1)

        torch.save(
            {
                "epoch": current_step,
                "total_steps": self.total_steps,
                "actor_state_dict": self.agent.actor.state_dict(),
                "critic_state_dict": self.agent.critic.state_dict(),
                "optimizer_state_dict": self.agent.optimizer.state_dict(),
                # TODO save/load reward normalization mean, std, count
            },
            os.path.join(version_dir, "checkpoint.pt"),
        )

        if save_actor_jit:
            traced_actor = th.jit.trace(self.agent.actor, self.jit_tracer)
            torch.jit.save(traced_actor, os.path.join(version_dir, "jit_policy.jit"))

    def freeze_policy(self, frozen_iterations=100):
        """
        Freeze policy network to allow value network to settle. Useful with pretrained policy networks.

        Note that network weights will not be transmitted when frozen.

        :param frozen_iterations: how many iterations the policy update will remain unchanged
        """

        print("-------------------------------------------------------------")
        print("Policy Weights frozen for " + str(frozen_iterations) + " iterations")
        print("-------------------------------------------------------------")

        self.frozen_iterations = frozen_iterations

        self._saved_lr = self.agent.optimizer.param_groups[0]["lr"]
        self.agent.optimizer.param_groups[0]["lr"] = 0


# adapted from AechPro distrib-rl by AechPro and SomeRando
# https://github.com/AechPro/distrib-rl/blob/main/distrib_rl/policy_optimization/learning_rate_controllers/pid_learning_rate_controller.py
class PIDLearningRateController(object):
    def __init__(
            self,
            target=0.025,
            kp=0.1,
            ki=0,
            kd=0,
            min_output=1e-7,
            max_output=1,
            max_clip_error=0.05,
    ):
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.max_clip_error = max_clip_error

        self.last_error = 0
        self.integral = 0

    def adjust(self, mean_clip, current_lr):
        # clip_target = self.clip_target

        # mean_lr = 0
        # clip mean_clip so it can't fall too quickly
        mean_clip = min(mean_clip, self.max_clip_error)

        error = self.target - mean_clip

        proportional = error * self.kp

        derivative = (error - self.last_error) * self.ki

        self.integral += error * self.kd

        adjustment = proportional + derivative + self.integral

        self.last_error = error

        return min(max(current_lr + adjustment, self.min_output), self.max_output)

        # if hasattr(optimizer, "torch_optimizer"):
        #     mean_lr = 0
        #     n = 0
        #     for group in optimizer.torch_optimizer.param_groups:
        #         if "lr" in group.keys():
        #             group["lr"] = min(max(group["lr"] + adjustment, min_lr), max_lr)
        #             mean_lr += group["lr"]
        #             n += 1
        #     return mean_lr / n
        # else:
        #     optimizer.step_size = min(
        #         max(optimizer.step_size + adjustment, min_lr), max_lr
        #     )
        #     return optimizer.step_size
