from typing import List

import numpy as np
import torch
from rlgym_sim.gym import Gym
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward
from rlgym_sim.utils.state_setters import DefaultState
from tqdm import tqdm

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter
from rocket_learn.utils.util import make_python_state
from rocket_learn.utils.truncated_condition import TruncatedCondition
import pickle


def generate_episode(env: Gym, policies, eval_setter=DefaultState(), evaluate=False, scoreboard=None,
                     progress=False, rust_sim=False, infinite_boost_odds=0,
                     send_gamestates=False,
                     ) -> (
        List[ExperienceBuffer], int):
    """
    create experience buffer data by interacting with the environment(s)
    """
    if progress:
        progress = tqdm(unit=" steps")
    else:
        progress = None
    # TODO allow evaluate with rust later by providing these values or bypassing
    if evaluate:  # Change setup temporarily to play a normal game (approximately)
        from game_condition import GameCondition  # tools is an optional dependency
        terminals = env._match._terminal_conditions  # noqa
        reward = env._match._reward_fn  # noqa
        game_condition = GameCondition(seconds_per_goal_forfeit=10 * 3,  # noqa
                                       max_overtime_seconds=300,
                                       max_no_touch_seconds=30)  # noqa
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

    if scoreboard is not None:
        random_resets = scoreboard.random_resets
        scoreboard.random_resets = not evaluate
    # TODO make rust binding
    if not rust_sim:
        observations, info = env.reset(return_info=True)
    else:
        observations = env.reset(len(policies) // 2, infinite_boost_odds=infinite_boost_odds)
        info = {'result': 0.0}
        # observations = env.reset()

    result = 0
    from_python_actions = np.load("python_actions.npy")
    step_count = 0
    last_state = info['state'] if not rust_sim else None  # game_state for obs_building of other agents

    latest_policy_indices = [0 if isinstance(p, HardcodedAgent) else 1 for p in policies]
    # rollouts for all latest_policies
    if not rust_sim:
        rollouts = [
            ExperienceBuffer(infos=[info])
            for _ in range(sum(latest_policy_indices))
        ]
    else:
        rollouts = [
            ExperienceBuffer()
            for _ in range(sum(latest_policy_indices))
        ]
    python_actions = []
    b = o = 0
    with torch.no_grad():
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
            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]
            # this doesn't seem to be working, due to different locations of policy?
            if not isinstance(policies[0], HardcodedAgent) and all(policy == policies[0] for policy in policies):
                # if True:
                policy = policies[0]
                if isinstance(observations[0], tuple):
                    obs = tuple(np.concatenate([obs[i] for obs in observations], axis=0)
                                for i in range(len(observations[0])))
                else:
                    # obs = np.concatenate(observations, axis=0)
                    # expand just for testing, do in obs builder normally?
                    obs = np.array(observations)
                    # obs = observations
                # mirror = obs[-1]
                # obs = obs[:-1]
                dist = policy.get_action_distribution(obs)
                # to_dump = (obs, dist)
                # fh = open("obs-dist.pkl", "ab")
                # pickle.dump(to_dump, fh)
                action_indices = policy.sample_action(dist)
                log_probs = policy.log_prob(dist, action_indices)
                actions = policy.env_compatible(action_indices)

                all_indices.extend(list(action_indices.numpy()))
                all_actions.extend(list(actions))
                all_log_probs.extend(list(log_probs.numpy()))
            else:
                index = 0
                for policy, obs in zip(policies, observations):
                    if isinstance(policy, HardcodedAgent):
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
                        action_indices = policy.sample_action(dist)[0]
                        log_probs = policy.log_prob(dist, action_indices).item()
                        actions = policy.env_compatible(action_indices)

                        all_indices.append(action_indices.numpy())
                        all_actions.append(actions)
                        all_log_probs.append(log_probs)

                    else:
                        print(str(type(policy)) + " type use not defined")
                        assert False

                    index += 1

            # to allow different action spaces, pad out short ones to longest length (assume later unpadding in parser)
            # length = max([a.shape[0] for a in all_actions])
            # padded_actions = []
            # for a in all_actions:
            #     action = np.pad(a.astype('float64'), (0, length - a.size), 'constant', constant_values=np.NAN)
            #     padded_actions.append(action)
            #
            # all_actions = padded_actions
            # TEST OUT ABOVE TO DEAL WITH VARIABLE LENGTH

            all_actions = np.vstack(all_actions)
            old_obs = observations

            # put the mirror back so I can handle it in the parser in Rust (hopefully)
            if len(mirror) > 0:
                all_actions = np.column_stack((all_actions, mirror))
            if not rust_sim:
                # print(f"python index: {all_actions}")
                # print(all_actions)
                step_count += 1
                python_actions.append(all_actions)
                observations, rewards, done, info = env.step(all_actions)
                dist = np.linalg.norm(observations[0][51:54])
                vel_diff = np.linalg.norm(observations[0][54:57])
                print(f"{step_count}: {all_actions[0]}:{all_actions[1]}: {dist}\t{vel_diff}")
            else:
                # print(f"python index: {all_actions}")
                all_actions = from_python_actions[step_count]
                step_count += 1
                observations, rewards, done, info, state = env.step(all_actions)
                dist = np.linalg.norm(observations[0][51:54])
                vel_diff = np.linalg.norm(observations[0][54:57])
                print(f"{step_count}: {all_actions[0]}:{all_actions[1]}: {dist}\t{vel_diff}")
                # state is a f32 vector of the state
                if send_gamestates:
                    info['state'] = make_python_state(state)
                else:
                    info['state'] = None
            # print(f"rewards in python are {rewards}")

            # TODO: add truncated eventually?
            # truncated = False
            # for terminal in env._match._terminal_conditions:  # noqa
            #     if isinstance(terminal, TruncatedCondition):
            #         truncated |= terminal.is_truncated(info["state"])

            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]

            # prune data that belongs to old agents
            old_obs = [a for i, a in enumerate(old_obs) if latest_policy_indices[i] == 1]
            all_indices = [d for i, d in enumerate(all_indices) if latest_policy_indices[i] == 1]
            rewards = [r for i, r in enumerate(rewards) if latest_policy_indices[i] == 1]
            all_log_probs = [r for i, r in enumerate(all_log_probs) if latest_policy_indices[i] == 1]

            assert len(old_obs) == len(all_indices), str(len(old_obs)) + " obs, " + str(len(all_indices)) + " ind"
            assert len(old_obs) == len(rewards), str(len(old_obs)) + " obs, " + str(len(rewards)) + " ind"
            assert len(old_obs) == len(all_log_probs), str(len(old_obs)) + " obs, " + str(len(all_log_probs)) + " ind"
            assert len(old_obs) == len(rollouts), str(len(old_obs)) + " obs, " + str(len(rollouts)) + " ind"

            # Might be different if only one agent?
            if not evaluate:  # Evaluation matches can be long, no reason to keep them in memory
                for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, all_indices, rewards, all_log_probs):
                    # exp_buf.add_step(obs, act, rew, done + 2 * truncated, log_prob, info)
                    # if not rust_sim:
                    exp_buf.add_step(obs, act, rew, done, log_prob, info)
                    # else:
                    #     exp_buf.add_step(obs, act, rew, done, log_prob, [])
                    # print(f"actions going to buffer are {act} and rewards are {rew}")
            # TODO skipping for now for rust to not hack on _match
            if progress is not None:
                progress.update()
                igt = progress.n * env._match._tick_skip / 120  # noqa
                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                if evaluate:
                    prog_str += f", BLUE {b} - {o} ORANGE"
                progress.set_postfix_str(prog_str)

            if done:  # or truncated:
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
                    # else:
                    #     observations = env.reset()

            last_state = info['state'] if send_gamestates else None

    if scoreboard is not None:
        scoreboard.random_resets = random_resets  # noqa Checked above

    if evaluate:
        if isinstance(env._match._state_setter, DynamicGMSetter):  # noqa
            env._match._state_setter.setter = state_setter  # noqa
        else:
            env._match._state_setter = state_setter  # noqa
        env._match._terminal_conditions = terminals  # noqa
        env._match._reward_fn = reward  # noqa
        return result

    if progress is not None:
        progress.close()

    if not rust_sim:
        np.save("python_actions.npy", python_actions)
    exit()
    return rollouts, result

