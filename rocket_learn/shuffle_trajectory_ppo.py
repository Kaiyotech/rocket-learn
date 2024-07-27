import json
import os
import time
from json import JSONDecodeError
from typing import Iterator, List, Tuple, Union

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
from rocket_learn.ppo import PIDLearningRateController
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator

torch.set_default_dtype(torch.float32)


class ShuffleTrajectoryPPO:
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
        self.is_selector = rollout_generator.selector_skip_k is not None  # noqa
        self.enable_ep_action_dist_calcs = (
            rollout_generator.enable_ep_action_dist_calcs  # noqa
        )  # noqa
        assert (
            minibatch_size is None
        ), "Shuffle Trajectory PPO does not use minibatching"
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
                total_size = 0
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
                            total_size += rollout.size()
                            # progress.update(rollout.size())
                            yield rollout
                    except StopIteration:
                        return
                perc_old_data = old_data / (old_data + new_data)
                self.logger.log(
                    {
                        "ppo/%old_data": perc_old_data,
                        "ppo/wasted_data": wasted_data,
                        "ppo/unlearnable_count": total_size - size,
                    },
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
        trajectory_batch_size = len(trajectory_actions)
        selector_choice_probs = self.rollout_generator.selector_skip_probability_table[
            :trajectory_batch_size, :trajectory_batch_size
        ]

        trajectory_actions = trajectory_actions.to(self.device)
        selector_choice_probs = th.as_tensor(
            selector_choice_probs, device=self.device, dtype=th.float32
        )

        with torch.autograd.graph.save_on_cpu():
            dist = self.agent.actor.get_action_distribution(trajectory_observations)
            dist_entropy = dist.entropy()[:, 0]
            dist_probs = dist.probs[:, 0, :]
            log_prob_tensor = th.log(
                th.matmul(selector_choice_probs, dist_probs).gather(
                    1, trajectory_actions
                )
            )[:, 0]
            entropy_tensor = th.matmul(selector_choice_probs, dist_entropy)
        return log_prob_tensor, entropy_tensor, dist_entropy

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
        return log_prob, entropy, dist

    @staticmethod
    @numba.njit
    def _calculate_advantages_numba(
        rewards,
        values,
        next_values,
        step_non_terminating,
        step_non_done,
        gamma,
        gae_lambda,
    ):
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0.0
        size = len(advantages)
        for step in range(size - 1, -1, -1):
            v_target = (
                rewards[step] + gamma * next_values[step] * step_non_terminating[step]
            )
            delta = v_target - values[step]
            last_gae_lam = (
                delta + gamma * gae_lambda * step_non_done[step] * last_gae_lam
            )
            advantages[step] = last_gae_lam
        return advantages

    def calculate(self, buffers_iterator: Iterator[ExperienceBuffer], iteration):
        """
        Calculate loss and update network
        """
        ep_rewards = []
        ep_steps = []
        action_count = np.asarray([0] * self.num_actions)
        action_changes = 0

        buffers = list(buffers_iterator)
        n_buffers = len(buffers)
        obs_is_tuple = isinstance(buffers[0].observations[0], tuple) or isinstance(
            buffers[0].observations[0], list
        )
        # ---------- START LOGGING BLOCK ----------
        for buffer in buffers:

            actions = np.stack(buffer.actions)
            rewards = np.stack(buffer.rewards)
            size = rewards.shape[0]
            ep_rewards.append(rewards.sum())
            ep_steps.append(size)

            if self.action_selection_dict is not None:
                flat_actions = actions[:, 0].flatten()
                unique, counts = np.unique(flat_actions, return_counts=True)
                for i, value in enumerate(unique):
                    action_count[value] += counts[i]
                action_changes += (np.diff(flat_actions) != 0).sum()

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
        # ---------- END LOGGING BLOCK ----------

        tot_loss = 0
        tot_policy_loss = 0
        tot_entropy_loss = 0
        tot_step_entropy_loss = 0
        tot_value_loss = 0
        total_kl_div = 0
        tot_clipped = 0

        if self.kl_models_weights is not None:
            tot_kl_other_models = np.zeros(len(self.kl_models_weights))
            tot_kl_coeffs = np.zeros(len(self.kl_models_weights))

        n = 0

        # if self.jit_tracer is None:
        #     self.jit_tracer = obs_tensor[0].to(self.device)

        print("Training network...")

        if self.frozen_iterations > 0:
            print("Policy network frozen, only updating value network...")

        precompute = torch.cat(
            [param.view(-1) for param in self.agent.actor.parameters()]
        )
        t0 = time.perf_counter_ns()
        self.agent.optimizer.zero_grad(set_to_none=self.zero_grads_with_none)
        n_minibatches_per_batch = self.batch_size // self.minibatch_size
        for _ in range(self.epochs):
            for _ in range(n_minibatches_per_batch):
                minibatch_obs_tensors = []
                minibatch_actions_tensors = []
                minibatch_log_probs_tensors = []
                minibatch_rewards_list = []
                minibatch_dones_list = []
                minibatch_learnable_mask_list = []
                next_value_pred_indices = []
                index_order = th.randperm(n_buffers)
                trajectory_minibatch_cur_timesteps = 0
                for buffer_index in index_order:
                    buffer = buffers[buffer_index]
                    buf_size = buffer.size()
                    if (
                        buf_size + trajectory_minibatch_cur_timesteps
                        > self.minibatch_size
                    ):
                        break
                    if obs_is_tuple:
                        transposed = tuple(zip(*buffer.observations))
                        obs_tensor = tuple(
                            torch.from_numpy(np.vstack(t)).float().to(self.device)
                            for t in transposed
                        )
                    else:  # use just this section for advanced obs
                        obs_tensor = (
                            th.from_numpy(np.vstack(buffer.observations))
                            .float()
                            .to(self.device)
                        )

                    actions_tensor = th.from_numpy(np.stack(buffer.actions)).to(
                        self.device
                    )
                    log_probs_tensor = th.from_numpy(np.stack(buffer.log_probs)).to(
                        self.device
                    )
                    rewards = np.stack(buffer.rewards)
                    dones = np.stack(buffer.dones)
                    learnable_mask = np.stack(buffer.learnable)

                    minibatch_obs_tensors.append(obs_tensor)
                    minibatch_actions_tensors.append(actions_tensor)
                    minibatch_log_probs_tensors.append(log_probs_tensor)
                    minibatch_rewards_list.append(rewards.astype(np.float32))
                    minibatch_dones_list.append(dones)
                    minibatch_learnable_mask_list.append(learnable_mask)
                    # Explained below
                    next_value_pred_indices += [
                        trajectory_minibatch_cur_timesteps + idx
                        for idx in range(1, buf_size)
                    ]
                    next_value_pred_indices.append(next_value_pred_indices[-1])
                    trajectory_minibatch_cur_timesteps += buf_size
                # A time step is a tuple where, for a given state:
                # - the obs is constructed from the state
                # - the action is the result of the obs
                # - the reward is the reward from the state after having taken the action
                # - the done is the result of the termination / truncation conditions on the state after having taken the action
                # If the next time step is terminated, then there is no action taken on the next time step.
                # We don't use the value function, instead we just use zero as the expected returns for the next state
                # Otherwise, we do use it.
                # When the next time step is terminated or truncated, *we don't have that observation*.
                # There is no time step in here that includes that observation.
                # The advantages calculation wants the value function's result for the following timestep,
                # including for the last time step we have.
                # For a given time step, the value prediction we want to use for it is the
                # value prediction for the next observation. We don't have the last observation,
                # so the best we can do is just use the value prediction for the last time step we have
                # twice.
                # We only want to use the last value prediction in a trajectory if the trajectory was
                # truncated anyway, so the impact of this is pretty negligible.

                # ---------- START BATCH QUANTITY DEFINITIONS ----------
                if obs_is_tuple:
                    transposed = zip(*minibatch_obs_tensors)
                    minibatch_obs_tensor = tuple(th.cat(t).float() for t in transposed)
                else:
                    minibatch_obs_tensor = th.cat(minibatch_obs_tensors).float()
                minibatch_value_preds_tensor = self.agent.critic(
                    minibatch_obs_tensor
                ).flatten()
                minibatch_value_preds = (
                    minibatch_value_preds_tensor.detach().cpu().numpy()
                )
                minibatch_next_value_preds = minibatch_value_preds[
                    next_value_pred_indices
                ]
                minibatch_dones = np.concatenate(minibatch_dones_list)
                minibatch_rewards = np.concatenate(minibatch_rewards_list)
                minibatch_log_probs_tensor = th.cat(minibatch_log_probs_tensors)
                # We want to use the next value pred for a time step only if the
                # done flag for the time step is 2 (this means the episode was truncated and not
                # terminated at the state resulting from having taken the action)
                # or the done flag for the time step is 0 (this means the episode was neither
                # truncated nor terminated at the state resulting from having taken the action)
                step_non_terminating = (minibatch_dones != 1) * (minibatch_dones != 3)
                # Since there are multiple trajectories in one array here, we want to ensure
                # that the gae lambda from the start of one trajectory isn't used
                # at the end of the next trajectory. We want to use the last gae lambda value
                # only if the done flag for the time step is 0
                step_non_done = minibatch_dones == 0
                advantages = self._calculate_advantages_numba(
                    minibatch_rewards,
                    minibatch_value_preds,
                    minibatch_next_value_preds,
                    step_non_terminating,
                    step_non_done,
                    self.gamma,
                    self.gae_lambda,
                )
                minibatch_advantages_tensor = th.from_numpy(advantages).to(self.device)
                minibatch_returns_tensor = (
                    minibatch_advantages_tensor + minibatch_value_preds_tensor.detach()
                )
                minibatch_actions_tensor = th.cat(minibatch_actions_tensors).to(
                    self.device
                )
                if self.is_selector and self.enable_ep_action_dist_calcs:
                    cur_policy_trajectories_log_probs = []
                    cur_policy_trajectories_entropy = []
                    cur_policy_trajectories_step_entropy = []
                    for obs_tensor, actions_tensor in zip(
                        minibatch_obs_tensors, minibatch_actions_tensors
                    ):
                        (
                            cur_policy_trajectory_log_probs,
                            cur_policy_trajectory_entropy,
                            cur_policy_trajectory_step_entropy,
                        ) = self.evaluate_actions_selector(obs_tensor, actions_tensor)
                        cur_policy_trajectories_log_probs.append(
                            cur_policy_trajectory_log_probs
                        )
                        cur_policy_trajectories_entropy.append(
                            cur_policy_trajectory_entropy
                        )
                        cur_policy_trajectories_step_entropy.append(
                            cur_policy_trajectory_step_entropy
                        )
                    minibatch_cur_policy_log_probs_tensor = th.cat(
                        cur_policy_trajectories_log_probs
                    )
                    minibatch_cur_policy_entropy_tensor = th.cat(
                        cur_policy_trajectories_entropy
                    )
                    minibatch_cur_policy_step_entropy_tensor = th.cat(
                        cur_policy_trajectories_step_entropy
                    )
                else:
                    (
                        minibatch_cur_policy_log_probs_tensor,
                        minibatch_cur_policy_entropy_tensor,
                        minibatch_dist,
                    ) = self.evaluate_actions(
                        minibatch_obs_tensor, minibatch_actions_tensor
                    )
                # ---------- END BATCH QUANTITY DEFINITIONS ----------

                ret = minibatch_returns_tensor
                old_log_prob = minibatch_log_probs_tensor
                log_prob = minibatch_cur_policy_log_probs_tensor
                entropy = -th.mean(minibatch_cur_policy_entropy_tensor)
                entropy_step = -th.mean(minibatch_cur_policy_step_entropy_tensor)
                diff_log_prob = log_prob - old_log_prob
                # stabilize the ratio for small log prob
                ratio = torch.where(
                    diff_log_prob.abs() < 0.00005,
                    1 + diff_log_prob,
                    torch.exp(diff_log_prob),
                )
                if self.extra_prints:
                    print(
                        f"ratio.min is {ratio.min()}  max is {ratio.max()}  mean is {ratio.mean()}"
                    )
                values_pred = minibatch_value_preds_tensor
                values_pred = th.squeeze(values_pred)
                adv = minibatch_advantages_tensor
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

                entropy_loss = entropy

                kl_loss = th.as_tensor(0, dtype=th.float32, device=self.device)
                if self.kl_models_weights is not None:
                    for k, (model, kl_coef, half_life) in enumerate(
                        self.kl_models_weights
                    ):
                        if half_life is not None:
                            kl_coef *= 0.5 ** (self.total_steps / half_life)
                        with torch.no_grad():
                            minibatch_dist_other = model.get_action_distribution(
                                minibatch_obs_tensor
                            )
                        minibatch_div = kl_divergence(
                            minibatch_dist_other, minibatch_dist
                        ).mean()
                        tot_kl_other_models[k] += minibatch_div
                        tot_kl_coeffs[k] = kl_coef
                        kl_loss += kl_coef * minibatch_div

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + kl_loss
                ) * (trajectory_minibatch_cur_timesteps / self.batch_size)

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
                    if obs_is_tuple:
                        for j in range(len(minibatch_obs_tensor)):
                            print(
                                f"\tObs[{j}] has inf:",
                                not minibatch_obs_tensor[j].isfinite().all(),
                            )
                    else:
                        print(
                            "\tObs has inf:", not minibatch_obs_tensor.isfinite().all()
                        )
                    continue

                loss.backward()

                # Unbiased low variance KL div estimator from http://joschu.net/blog/kl-approx.html
                total_kl_div += th.mean((ratio - 1) - (log_prob - old_log_prob)).item()
                tot_loss += loss.item()
                tot_policy_loss += policy_loss.item()
                tot_entropy_loss += entropy_loss.item()
                tot_step_entropy_loss += entropy_step.item()
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
            "ppo/entropy_step_loss": tot_step_entropy_loss / n,
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
        assert not save_actor_jit, "not implemented"
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
