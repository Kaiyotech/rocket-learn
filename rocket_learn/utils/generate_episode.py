import copy
import os
import random
import string
from typing import List

import numpy as np
import pretrained_agents.Opti.Opti_submodel
import torch
from rlgym_sim.gym import Gym
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward
from rlgym_sim.utils.state_setters import DefaultState
from tqdm import tqdm

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter
from rocket_learn.utils.truncated_condition import TruncatedCondition
from rocket_learn.utils.util import gamestate_to_replay_array, make_python_state

# import pickle

torch.set_default_dtype(torch.float32)

def generate_episode(
    env: Gym,
    policies,
    eval_setter=DefaultState(),
    evaluate=False,
    scoreboard=None,
    progress=False,
    rust_sim=False,
    infinite_boost_odds=0,
    streamer=False,
    send_gamestates=False,
    reward_stage=0,
    gather_data=False,
    selector=False,
    ngp_reward=None,
    selector_parser=None,
    selector_skip_k=None,
    min_learnable_action_prob=None,
    selector_skip_probability_table=None,
    enable_ep_action_dist_calcs=False,
    ngp_action_parser=None,
    num_submodels=None,
) -> (List[ExperienceBuffer], int):
    """
    create experience buffer data by interacting with the environment(s)
    """
    if progress:
        progress = tqdm(unit=" steps")
    else:
        progress = None
    # TODO allow evaluate with rust later by providing these values or bypassing
    if evaluate:  # Change setup temporarily to play a normal game (approximately)
        from from_rlgym_or_tools.game_condition import (
            GameCondition,  # tools is an optional dependency
        )

        terminals = env._match._terminal_conditions  # noqa
        reward = env._match._reward_fn  # noqa
        game_condition = GameCondition(
            seconds_per_goal_forfeit=10 * 3,  # noqa
            max_overtime_seconds=300,
            max_no_touch_seconds=30,
        )  # noqa
        env._match._terminal_conditions = [game_condition]  # noqa
        if isinstance(env._match._state_setter, DynamicGMSetter):  # noqa
            state_setter = env._match._state_setter.setter  # noqa
            env._match._state_setter.setter = eval_setter  # noqa
            env.update_settings(boost_consumption=1)  # remove infinite boost
        else:
            state_setter = env._match._state_setter  # noqa
            env._match._state_setter = eval_setter  # noqa
            env.update_settings(boost_consumption=1)  # remove infinite boost

        env._match._reward_fn = ConstantReward()  # noqa Save some cpu cycles

    if ngp_reward is not None:
        accumulated_old_obs = []
        accumulated_all_indices = []
        accumulated_rewards = []
        accumulated_all_log_probs = []
        accumulated_states = []
        accumulated_dones = []
        accumulated_infos = []
        accumulated_actions = []
        send_gamestates = True

    if scoreboard is not None:
        random_resets = scoreboard.random_resets
        scoreboard.random_resets = not evaluate
    if any(isinstance(policy, HardcodedAgent) for policy in policies):
        # we have pretrained models that need gamestate
        send_gamestates = True
    if gather_data:  # gathering data requires the gamestate
        send_gamestates = True
        file_name = "gather_data\\" + "".join(
            random.choice(string.ascii_lowercase) for i in range(16)
        )
        to_save = []
    if not rust_sim:
        observations, info = env.reset(return_info=True)
    else:
        observations, state = env.reset(
            len(policies) // 2,
            infinite_boost_odds=infinite_boost_odds,
            send_gamestate=send_gamestates,
            reward_stage=reward_stage,
        )
        info = {"result": 0.0}
        if send_gamestates:
            info["state"] = make_python_state(state)
        else:
            info["state"] = None
        # observations = env.reset()
    sliders = None
    if streamer:
        slider_string = ""
        scoreboard_string = ""
        # get sliders from file on reset, None if doesn't exist
        sliders = get_sliders()

    result = 0

    last_state = info["state"]  # game_state for obs_building of other agents

    latest_policy_indices = [
        0 if isinstance(p, HardcodedAgent) else 1 for p in policies
    ]
    # rollouts for all latest_policies
    if not rust_sim:
        rollouts = [
            ExperienceBuffer(infos=[info]) for _ in range(sum(latest_policy_indices))
        ]
    else:
        rollouts = [ExperienceBuffer() for _ in range(sum(latest_policy_indices))]

    b = o = 0
    if gather_data:
        data_ticks_passed = 30
        gather_data_ticks = random.uniform(15, 45)
    # fh_pickle = open("testing_state.pkl", 'wb')
    for i, value in enumerate(latest_policy_indices):
        if value == 1 and policies[i].shape[0] < 50:
            num_submodels = policies[i].shape[0]
            break
    if selector:
        send_gamestates = True  # needed for evals and everything
        if not evaluate and enable_ep_action_dist_calcs:
            if selector_skip_probability_table is not None:
                max_size = len(selector_skip_probability_table)
                episode_action_distributions = [
                    torch.zeros((max_size, num_submodels), dtype=torch.float32) for _ in policies
                ]
            else:
                episode_action_distributions = [
                    torch.as_tensor([], dtype=torch.float32) for _ in policies
                ]
            steps_since_episode_start = 0
    # eval with no selector
    selector_eval_only_others = False
    if num_submodels is None:
        selector_eval_only_others = selector
        selector = False
    with torch.no_grad():
        tick = [0] * len(policies)
        do_selector = [True] * len(policies)
        last_actions = [None] * len(policies)
        if selector:
            last_model_action = torch.tensor(
                [[random.choice(range(num_submodels))] for _ in range(len(policies))],
                dtype=torch.long,
            )
        min_learnable_action_logprob = (
            None
            if min_learnable_action_prob is None
            else torch.log(torch.as_tensor(min_learnable_action_prob))
        )
        while True:
            all_indices = []
            all_actions = []
            all_log_probs = []
            mirror = []
            # need to remove the mirror from the end here instead of later so it doesn't make it to ppo
            if len(observations[0]) == 3:
                for i, observation in enumerate(observations):
                    mirror.append(observation[2])
                    observations[i] = observations[i][:-1]
            actual_prev_actions = [None] * len(observations)
            if selector:
                # need to update the model action with last model action one-hot
                # and remove the actual prev actions but save them for later to give to submodels
                for i, observation in enumerate(observations):
                    one_hot = np.zeros(num_submodels)
                    one_hot[last_model_action[i][0]] = 1
                    observations[i][0][:, 199:][0] = one_hot

                    # 37 through 45 are prev action
                    actual_prev_actions[i] = observations[i][0][:, 37:45][0]

            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]
            # this doesn't seem to be working, due to different locations of policy?
            if not isinstance(policies[0], HardcodedAgent) and all(
                policy == policies[0] for policy in policies
            ):
                # if True:
                policy = policies[0]
                if isinstance(observations[0], tuple):
                    obs = tuple(
                        np.concatenate([obs[i] for obs in observations], axis=0)
                        for i in range(len(observations[0]))
                    )
                else:
                    # obs = np.concatenate(observations, axis=0)
                    # expand just for testing, do in obs builder normally?
                    obs = np.array(observations)
                    # obs = observations
                # mirror = obs[-1]
                # obs = obs[:-1]

                dist = policy.get_action_distribution(obs)
                if selector and not evaluate and enable_ep_action_dist_calcs:
                    for index, probs in enumerate(dist.probs[:, 0, :]):
                        episode_action_distributions[index][
                            steps_since_episode_start
                        ] = probs
                # to_dump = (obs, dist)
                # fh = open("obs-dist.pkl", "ab")
                # pickle.dump(to_dump, fh)

                if not selector:
                    action_indices = policy.sample_action(dist)
                    actions = policy.env_compatible(action_indices)
                else:
                    potential_action_indices = policy.sample_action(dist)
                    action_indices = []
                    for idx, should_use in enumerate(do_selector):
                        if should_use:
                            action_indices.append([potential_action_indices[idx]])
                            last_model_action[idx] = potential_action_indices[idx]
                        else:
                            action_indices.append([last_model_action[idx]])
                    action_indices = torch.as_tensor(action_indices)

                    # need to remove the prev model actions and add the prev actions
                    obs[0][:, 37:45] = np.array(actual_prev_actions)
                    obs = (obs[0][:, :-num_submodels], obs[1])
                    actions = selector_parser.parse_actions(
                        action_indices, last_state, obs
                    )

                all_indices.extend(list(action_indices.numpy()))
                all_actions.extend(list(actions))
                if selector and not evaluate and enable_ep_action_dist_calcs:
                    log_probs = torch.zeros(
                        len(episode_action_distributions), dtype=torch.float32
                    )
                    for idx, episode_action_distribution in enumerate(
                        episode_action_distributions
                    ):
                        log_probs[idx] = torch.as_tensor(
                            calculate_action_log_prob(
                                steps_since_episode_start,
                                episode_action_distribution,
                                action_indices[idx],
                                selector_skip_probability_table,
                            )
                        )
                else:
                    log_probs = policy.log_prob(dist, action_indices)

                all_log_probs.extend(list(log_probs.numpy()))
            else:
                index = 0
                for policy, obs in zip(policies, observations):
                    # if isinstance(obs, tuple):
                    #     obs = tuple(np.concatenate([obs[i] for obs in observations], axis=0)
                    #                 for i in range(len(observations[0])))
                    if isinstance(policy, HardcodedAgent):
                        actions = None
                        # No reason to build another obs, just use the rust one that's already built
                        if isinstance(
                            policy, pretrained_agents.Opti.Opti_submodel.Submodel
                        ):
                            # remove model actions (if removed by selector) and add previous actions and put back mirror
                            if selector:
                                obs[0][:, 37:45] = np.array(actual_prev_actions[index])
                                obs = (obs[0][:, :-num_submodels], obs[1], mirror[index])
                            # gonna be selector obs, but no selector is present
                            elif selector_eval_only_others:
                                obs = (obs[0][:, :-num_submodels], obs[1], mirror[index])
                            else:
                                obs = (obs[0], obs[1], mirror[index])
                            actions = policy.act(obs, last_state, index)
                        else:
                            actions = policy.act(last_state, index)

                        # make sure output is in correct format
                        if not isinstance(observations, np.ndarray):
                            actions = np.array(actions)

                        # TODO: add converter that takes normal 8 actions into action space
                        # actions = env._match._action_parser.convert_to_action_space(actions)

                        all_indices.append(None)
                        all_actions.append(actions)
                        all_log_probs.append(None)

                    elif isinstance(policy, Policy):
                        dist = policy.get_action_distribution(obs)
                        if selector and not evaluate and enable_ep_action_dist_calcs:
                            episode_action_distributions[index][
                                steps_since_episode_start
                            ] = dist.probs
                        if not selector:
                            action_indices = policy.sample_action(dist)
                            actions = policy.env_compatible(action_indices)[0]
                        else:
                            if do_selector[index]:
                                action_indices = policy.sample_action(dist)
                                last_model_action[index] = action_indices

                            else:
                                action_indices = copy.deepcopy(last_model_action[index].unsqueeze(0))

                            # need to remove the prev model actions and add the prev actions
                            obs[0][:, 37:45] = np.array(actual_prev_actions[index])
                            obs = (obs[0][:, :-num_submodels], obs[1])
                            actions = selector_parser.parse_actions(
                                action_indices, last_state, obs
                            )

                        all_indices.extend(list(action_indices.numpy()))
                        all_actions.append(actions)
                        if selector and not evaluate and enable_ep_action_dist_calcs:
                            log_probs = torch.as_tensor(
                                calculate_action_log_prob(
                                    steps_since_episode_start,
                                    episode_action_distributions[index],
                                    action_indices[0],
                                    selector_skip_probability_table,
                                )
                            )
                        else:
                            log_probs = policy.log_prob(dist, action_indices)
                        all_log_probs.extend(list(log_probs.numpy().reshape(1)))

                    else:
                        print(str(type(policy)) + " type use not defined")
                        assert False

                    index += 1

                # pad because of pretrained
                length = max([a.shape[0] for a in all_actions])
                padded_actions = []
                for a in all_actions:
                    action = np.pad(
                        a.astype("float64"),
                        (0, length - a.size),
                        "constant",
                        constant_values=np.NAN,
                    )
                    padded_actions.append(action)

                all_actions = padded_actions

            # print out scoreboard and sliders if streamer
            if streamer:
                stream_obs = observations[0][0][0]
                if sliders is not None:
                    slider_range = range(61, 65)
                    stream_obs[slider_range] = sliders
                (slider_string, scoreboard_string) = print_stream_info(
                    slider_string, scoreboard_string, stream_obs
                )  # noqa
            if selector_skip_k is not None:
                for i in range(len(do_selector)):
                    if not isinstance(policies[i], HardcodedAgent):
                        do_selector[i] = do_selector_action(selector_skip_k[i], tick[i])
                        if policies[i].deterministic:
                            do_selector[i] = True
            else:
                do_selector = [True] * len(policies)
            for i in range(len(tick)):
                tick[i] = 0 if do_selector[i] else tick[i] + 1

            all_actions = np.vstack(all_actions)
            old_obs = observations

            # put the mirror back so I can handle it in the parser in Rust (hopefully)
            if len(mirror) > 0:
                all_actions = np.column_stack((all_actions, mirror))
            if not rust_sim:
                observations, rewards, done, info = env.step(all_actions)
            else:
                observations, rewards, done, info, state = env.step(all_actions)

                # state is a f32 vector of the state
                if send_gamestates:
                    info["state"] = make_python_state(state)
                    # print(f"state in python: {info['state']}")
                    # print(f"array in python: {state}")
                else:
                    info["state"] = None
                # pickle.dump((info['state'], gamestate_to_replay_array(info['state'])), fh_pickle)
                if gather_data:
                    data_ticks_passed += 1
                    if data_ticks_passed > gather_data_ticks:
                        data_ticks_passed = 0
                        gather_data_ticks = random.uniform(15, 45)
                        to_save.append(gamestate_to_replay_array(info["state"]))
            if selector and not evaluate and enable_ep_action_dist_calcs:
                steps_since_episode_start += 1

            # print(f"rewards in python are {rewards}")

            # truncated = False
            # for terminal in env._match._terminal_conditions:  # noqa
            #     if isinstance(terminal, TruncatedCondition):
            #         truncated |= terminal.is_truncated(info["state"])
            # truncated from rust is in info on key "truncated" and is either 0. or 1.
            truncated = False
            if rust_sim:
                truncated |= bool(info["truncated"])
            # if truncated:
            #     print("episode got truncated")

            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]

            # prune data that belongs to old agents
            old_obs = [
                a for i, a in enumerate(old_obs) if latest_policy_indices[i] == 1
            ]
            all_indices = [
                d for i, d in enumerate(all_indices) if latest_policy_indices[i] == 1
            ]
            rewards = [
                r for i, r in enumerate(rewards) if latest_policy_indices[i] == 1
            ]
            all_log_probs = [
                r for i, r in enumerate(all_log_probs) if latest_policy_indices[i] == 1
            ]

            assert len(old_obs) == len(all_indices), (
                str(len(old_obs)) + " obs, " + str(len(all_indices)) + " ind"
            )
            assert len(old_obs) == len(rewards), (
                str(len(old_obs)) + " obs, " + str(len(rewards)) + " ind"
            )
            assert len(old_obs) == len(all_log_probs), (
                str(len(old_obs)) + " obs, " + str(len(all_log_probs)) + " ind"
            )
            assert len(old_obs) == len(rollouts), (
                str(len(old_obs)) + " obs, " + str(len(rollouts)) + " ind"
            )

            # Might be different if only one agent?
            if (
                not evaluate and ngp_reward is None
            ):  # Evaluation matches can be long, no reason to keep them in memory
                for exp_buf, obs, act, rew, log_prob in zip(
                    rollouts, old_obs, all_indices, rewards, all_log_probs
                ):
                    exp_buf.add_step(
                        obs,
                        act,
                        rew,
                        done + 2 * truncated,
                        log_prob,
                        min_learnable_action_logprob is None
                        or min_learnable_action_logprob <= log_prob,
                        info,
                    )
                    # if not rust_sim:
                    # exp_buf.add_step(obs, act, rew, done, log_prob, info)
                    # else:
                    #     exp_buf.add_step(obs, act, rew, done, log_prob, [])
                    # print(f"actions going to buffer are {act} and rewards are {rew}")
            elif ngp_reward is not None:
                accumulated_old_obs.append(old_obs)
                accumulated_all_indices.append(all_indices)
                accumulated_rewards.append(rewards)
                accumulated_all_log_probs.append(all_log_probs)
                accumulated_states.append(info["state"])
                accumulated_dones.append(done + 2 * truncated)
                accumulated_infos.append(info)
                accumulated_actions.append(ngp_action_parser.parse_actions(all_actions, info["state"]))
            # TODO skipping for now for rust to not hack on _match
            if progress is not None:
                progress.update()
                igt = progress.n * env._match._tick_skip / 120  # noqa
                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                if evaluate:
                    prog_str += f", BLUE {b} - {o} ORANGE"
                progress.set_postfix_str(prog_str)

            if done or truncated:
                if gather_data:
                    np.save(file_name, np.asarray(to_save))
                if ngp_reward is not None:
                    updated_rewards = ngp_reward.add_ngp_rewards(
                        accumulated_rewards, accumulated_states, latest_policy_indices, accumulated_actions
                    )
                    for (
                        old_obs_list,
                        indices_list,
                        rewards_list,
                        log_probs_list,
                        done,
                        my_info,
                    ) in zip(
                        accumulated_old_obs,
                        accumulated_all_indices,
                        updated_rewards,
                        accumulated_all_log_probs,
                        accumulated_dones,
                        accumulated_infos,
                    ):
                        for exp_buf, old_obs, indices, rew, log_prob in zip(
                            rollouts,
                            old_obs_list,
                            indices_list,
                            rewards_list,
                            log_probs_list,
                        ):
                            exp_buf.add_step(
                                old_obs,
                                indices,
                                rew,
                                done,
                                log_prob,
                                min_learnable_action_logprob is None
                                or min_learnable_action_logprob <= log_prob,
                                my_info,
                            )
                # if not rust_sim:
                result += info["result"]
                if info["result"] > 0:
                    b += 1
                elif info["result"] < 0:
                    o += 1

                if not evaluate:
                    break
                elif game_condition.done:  # noqa
                    break
                else:
                    # if not rust_sim:
                    observations, info = env.reset(return_info=True)
                    tick = [0] * len(policies)
                    do_selector = [True] * len(policies)
                    # else:
                    #     observations = env.reset()

            last_state = info["state"] if send_gamestates else None

    if scoreboard is not None:
        scoreboard.random_resets = random_resets  # noqa Checked above

    if evaluate:
        if isinstance(env._match._state_setter, DynamicGMSetter):  # noqa
            env._match._state_setter.setter = state_setter  # noqa
        else:
            env._match._state_setter = state_setter  # noqa
        env._match._terminal_conditions = terminals  # noqa
        env._match._reward_fn = reward  # noqa
        scores = (info["state"].blue_score, info["state"].orange_score)
        return result, scores

    if progress is not None:
        progress.close()

    return rollouts, result


def print_stream_info(slider_string, scoreboard_string, obs):
    slider_range = range(61, 65)
    slider_names = ["Speed", "Aerial", "Aggressive", "Physical"]
    scoreboard_range = range(56, 61)
    stream_dir = (
        "C:\\Users\\kchin\\Code\\Kaiyotech\\Spectrum_play_redis\\stream_files\\"
    )
    time_remaining = round(obs[scoreboard_range.start] * 300)
    score_diff = obs[scoreboard_range.start + 1] * 5
    overtime = obs[scoreboard_range.start + 2]
    new_score_string = f"Time Remain: {time_remaining}\nScore Diff: {round(score_diff):+}\nOT: {bool(round(overtime))}"
    if scoreboard_string != new_score_string:
        scoreboard_string = new_score_string
        try:
            filename = os.path.join(stream_dir, "scoreboard.txt")
            with open(filename, "w") as f2:
                f2.write(f"{scoreboard_string}")
        except Exception as e:
            print(f"Error writing to file: {e}")

    new_slider_string = ""
    strings_i = 0
    for i in slider_range:
        value = obs[i]
        name = slider_names[strings_i]
        strings_i += 1
        new_slider_string += f"{name}: {value:.2f}"
        if i != slider_range.stop - 1:
            new_slider_string += "\n"
    if slider_string != new_slider_string:
        slider_string = new_slider_string
        try:
            filename = os.path.join(stream_dir, "sliders.txt")
            with open(filename, "w") as f2:
                f2.write(f"{slider_string}")
        except Exception as e:
            print(f"Error writing to file: {e}")
    return slider_string, scoreboard_string


def get_sliders():
    slider_file = "C:\\Users\\kchin\\Code\\Kaiyotech\\Spectrum_play_redis\\stream_files\\set_sliders_blue.txt"
    try:
        with open(slider_file, "r+") as fh:
            slider_values = fh.readline()
            slider_values = slider_values.split("!setslidersblue")[1].strip()
            if slider_values.startswith("used"):
                return None
            slider_values = slider_values.split(",")
            if len(slider_values) != 4:
                return None
            for i, value in enumerate(slider_values):
                slider_values[i] = min(max(float(value), -10), 10)  # noqa
            fh.seek(0, 0)
            fh.write("used\n")
            return slider_values
    except Exception as e:
        print(f"Error reading slider values: {e}")


def do_selector_action(selector_skip_k, tick) -> bool:
    p = 1 / (1 + (selector_skip_k * tick))
    return np.random.uniform() < 1 - p


def calculate_action_log_prob(
    episode_steps, episode_dists_probs, action_indices, selector_skip_probability_table
):
    if episode_steps == 0:
        assert action_indices.shape == (1,)
    return torch.log(
        torch.sum(
            episode_dists_probs[:, action_indices[0]]
            * selector_skip_probability_table[episode_steps]
        )
    )
