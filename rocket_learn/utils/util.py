from typing import Any

import numpy as np
import torch
import torch.distributions
from rlgym_sim.utils import math
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.obs_builders import AdvancedObs
from torch import nn


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SplitLayer(nn.Module):
    def __init__(self, splits=None):
        super().__init__()
        if splits is not None:
            self.splits = splits
        else:
            self.splits = (3,) * 5 + (2,) * 3

    def forward(self, x):
        return torch.split(x, self.splits, dim=-1)


# TODO AdvancedObs should be supported by default, use stack instead of cat
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> Any:
        return np.reshape(
            super(ExpandAdvancedObs, self).build_obs(player, state, previous_action),
            (1, -1),
        )


def probability_NvsM(team1_ratings, team2_ratings, env=None):
    from trueskill import global_env

    # Trueskill extension, source: https://github.com/sublee/trueskill/pull/17
    """Calculates the win probability of the first team over the second team.
    :param team1_ratings: ratings of the first team participants.
    :param team2_ratings: ratings of another team participants.
    :param env: the :class:`TrueSkill` object.  Defaults to the global
                environment.
    """
    if env is None:
        env = global_env()

    team1_mu = sum(r.mu for r in team1_ratings)
    team1_sigma = sum((env.beta**2 + r.sigma**2) for r in team1_ratings)
    team2_mu = sum(r.mu for r in team2_ratings)
    team2_sigma = sum((env.beta**2 + r.sigma**2) for r in team2_ratings)

    x = (team1_mu - team2_mu) / np.sqrt(team1_sigma + team2_sigma)
    probability_win_team1 = env.cdf(x)
    return probability_win_team1


def make_python_state(state_vals) -> GameState:
    player_len = 39
    boost_pad_length = 34
    state_vals = np.asarray(state_vals)
    state = GameState()
    state.game_type = state_vals[0]
    state.blue_score = state_vals[1]
    state.orange_score = state_vals[2]
    start = 3
    state.boost_pads = np.asarray(state_vals[start : start + boost_pad_length])
    start += boost_pad_length
    state.inverted_boost_pads = np.asarray(state_vals[start : start + boost_pad_length])
    start += boost_pad_length
    state.ball.position = np.asarray(
        (state_vals[start], state_vals[start + 1], state_vals[start + 2])
    )
    state.ball.linear_velocity = np.asarray(
        (state_vals[start + 3], state_vals[start + 4], state_vals[start + 5])
    )
    state.ball.angular_velocity = np.asarray(
        (state_vals[start + 6], state_vals[start + 7], state_vals[start + 8])
    )
    start += 9
    state.inverted_ball.position = np.asarray(
        (state_vals[start], state_vals[start + 1], state_vals[start + 2])
    )
    state.inverted_ball.linear_velocity = np.asarray(
        (state_vals[start + 3], state_vals[start + 4], state_vals[start + 5])
    )
    state.inverted_ball.angular_velocity = np.asarray(
        (state_vals[start + 6], state_vals[start + 7], state_vals[start + 8])
    )
    start += 9
    num_players = (len(state_vals) - start) // player_len
    for i in range(num_players):
        player = PlayerData()
        player.car_id = state_vals[start]
        player.team_num = state_vals[start + 1]
        start += 2
        player.car_data.position = np.asarray(
            (state_vals[start], state_vals[start + 1], state_vals[start + 2])
        )
        player.car_data.quaternion = np.asarray(
            (
                state_vals[start + 3],
                state_vals[start + 4],
                state_vals[start + 5],
                state_vals[start + 6],
            )
        )
        player.car_data.linear_velocity = np.asarray(
            (state_vals[start + 7], state_vals[start + 8], state_vals[start + 9])
        )
        player.car_data.angular_velocity = np.asarray(
            (state_vals[start + 10], state_vals[start + 11], state_vals[start + 12])
        )
        start += 13
        player.inverted_car_data.position = np.asarray(
            (state_vals[start], state_vals[start + 1], state_vals[start + 2])
        )
        player.inverted_car_data.quaternion = np.asarray(
            (
                state_vals[start + 3],
                state_vals[start + 4],
                state_vals[start + 5],
                state_vals[start + 6],
            )
        )
        player.inverted_car_data.linear_velocity = np.asarray(
            (state_vals[start + 7], state_vals[start + 8], state_vals[start + 9])
        )
        player.inverted_car_data.angular_velocity = np.asarray(
            (state_vals[start + 10], state_vals[start + 11], state_vals[start + 12])
        )
        start += 13
        player.car_data._euler_angles = math.quat_to_euler(player.car_data.quaternion)
        player.car_data._rotation_mtx = math.quat_to_rot_mtx(player.car_data.quaternion)
        player.inverted_car_data._euler_angles = math.quat_to_euler(
            player.inverted_car_data.quaternion
        )
        player.inverted_car_data._rotation_mtx = math.quat_to_rot_mtx(
            player.inverted_car_data.quaternion
        )
        player._has_computed_euler_angles = True
        player._has_computed_rot_mtx = True
        player.match_goals = state_vals[start]
        player.match_saves = state_vals[start + 1]
        player.match_shots = state_vals[start + 2]
        player.match_demolishes = state_vals[start + 3]
        player.boost_pickups = state_vals[start + 4]
        player.is_demoed = state_vals[start + 5]
        player.on_ground = state_vals[start + 6] > 0.0
        player.ball_touched = state_vals[start + 7] > 0.0
        player.has_jump = state_vals[start + 8] > 0.0
        player.has_flip = state_vals[start + 9] > 0.0
        player.boost_amount = state_vals[start + 10]
        start += 11
        state.players.append(player)

    state.players.sort(key=lambda p: p.car_id)

    return state


def gamestate_to_numpy_array(gamestate):
    # Initialize an empty list to hold the state values
    state_values = []

    # Add game type, scores, and boost pads
    state_values.extend(
        [gamestate.game_type, gamestate.blue_score, gamestate.orange_score]
    )
    state_values.extend(gamestate.boost_pads.tolist())
    state_values.extend(gamestate.inverted_boost_pads.tolist())

    # Add ball position, linear velocity, and angular velocity
    state_values.extend(gamestate.ball.position.tolist())
    state_values.extend(gamestate.ball.linear_velocity.tolist())
    state_values.extend(gamestate.ball.angular_velocity.tolist())

    # Add inverted ball position, linear velocity, and angular velocity
    state_values.extend(gamestate.inverted_ball.position.tolist())
    state_values.extend(gamestate.inverted_ball.linear_velocity.tolist())
    state_values.extend(gamestate.inverted_ball.angular_velocity.tolist())

    # Add player data
    # print(f"there are {len(gamestate.players)} players")
    for player in gamestate.players:
        # print(f"creating player {player.car_id}")
        state_values.append(player.car_id)
        state_values.append(player.team_num)
        state_values.extend(player.car_data.position.tolist())
        state_values.extend(player.car_data.quaternion.tolist())
        state_values.extend(player.car_data.linear_velocity.tolist())
        state_values.extend(player.car_data.angular_velocity.tolist())
        state_values.extend(player.inverted_car_data.position.tolist())
        state_values.extend(player.inverted_car_data.quaternion.tolist())
        state_values.extend(player.inverted_car_data.linear_velocity.tolist())
        state_values.extend(player.inverted_car_data.angular_velocity.tolist())
        state_values.append(player.match_goals)
        state_values.append(player.match_saves)
        state_values.append(player.match_shots)
        state_values.append(player.match_demolishes)
        state_values.append(player.boost_pickups)
        state_values.append(player.is_demoed)
        state_values.append(int(player.on_ground))
        state_values.append(int(player.ball_touched))
        state_values.append(int(player.has_jump))
        state_values.append(int(player.has_flip))
        state_values.append(player.boost_amount)

    # Convert the list to a NumPy array
    state_array = np.array(state_values)

    return state_array


def gamestate_to_replay_array(gamestate):
    # Initialize an empty list to hold the state values
    state_values = []

    # Add ball position, linear velocity, and angular velocity
    state_values.extend(gamestate.ball.position.tolist())
    state_values.extend(gamestate.ball.linear_velocity.tolist())
    state_values.extend(gamestate.ball.angular_velocity.tolist())

    # Add player data
    # print(f"there are {len(gamestate.players)} players")
    for player in gamestate.players:
        # print(f"creating player {player.car_id}")
        state_values.extend(player.car_data.position.tolist())
        euler = player.car_data.euler_angles()
        state_values.extend(euler.tolist())
        state_values.extend(player.car_data.linear_velocity.tolist())
        state_values.extend(player.car_data.angular_velocity.tolist())
        state_values.append(player.boost_amount)

    # Convert the list to a NumPy array
    state_array = np.array(state_values)

    return state_array


def generate_selector_skip_probability_table(episode_steps, selector_skip_k):
    if episode_steps == 0:
        return np.ones((1, 1))
    # First calculate probability that there was a selector action on a given step
    selector_action_step_probs = [1]
    prob_no_selector_action_taken_from_step_until_step = np.ones(
        (episode_steps + 1, episode_steps + 1)
    )
    for idx in range(1, episode_steps + 1):
        sum = 0
        for step, selector_action_step_prob in enumerate(selector_action_step_probs):
            # This is the probability that a selector action was taken on step step and no selector action was taken following step step (up until, but not necessarily including, step idx)
            prob_step_is_most_recent_prev_selector_action = (
                selector_action_step_prob
                * prob_no_selector_action_taken_from_step_until_step[step + 1, idx - 1]
            )
            # we add to the sum the probability that a selector action was taken on step step, no selector action was taken following step step until step idx, and then a selector action was taken on step idx
            sum += prob_step_is_most_recent_prev_selector_action * (
                1 - 1 / (1 + selector_skip_k * (idx - step))
            )
        prob_no_selector_action_taken_from_step_until_step[: idx + 1, idx] = (
            prob_no_selector_action_taken_from_step_until_step[: idx + 1, idx - 1]
            * (1 - sum)
        )
        selector_action_step_probs.append(sum)

    return prob_no_selector_action_taken_from_step_until_step


def calculate_prob_last_selector_action_at_step_for_step(
    step, selector_skip_probability_table
):
    # arr[i] is the probability that i was the last step that the selector made an action choice when the current step is step
    if step == 0:
        return np.ones(1)
    arr = np.zeros(step + 1)
    arr[0] = selector_skip_probability_table[1, step]
    for cur_step in range(1, step):
        arr[cur_step] = (
            1 - selector_skip_probability_table[cur_step, cur_step]
        ) * selector_skip_probability_table[cur_step + 1, step]
    arr[step] = 1 - selector_skip_probability_table[step, step]
    return arr


def calculate_prob_last_selector_action_at_step_for_steps(
    total_steps, selector_skip_probability_table
):
    # arr[i,j] is the probability that j was the last step that the selector made an action choice when the current step is i
    arr = np.zeros((total_steps + 1, total_steps + 1))
    arr[:, 0] = selector_skip_probability_table[1, : total_steps + 1]
    diagonal = np.diagonal(selector_skip_probability_table)
    for cur_step in range(1, total_steps):
        arr[:, cur_step] = (
            1 - selector_skip_probability_table[cur_step, cur_step]
        ) * selector_skip_probability_table[cur_step + 1, : total_steps + 1]
    arr[:, total_steps] = 1 - diagonal[: total_steps + 1]
    return arr * np.tril(np.ones((total_steps + 1, total_steps + 1)))
