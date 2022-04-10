from typing import Any
import numpy
import sys
import os

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.default_reward import DefaultReward
from rocket_learn.utils.mybots_statesets import WallDribble
from rocket_learn.utils.util import ExpandAdvancedObs
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker

from Constants import *
from rewards import anneal_rewards_fn
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter


if __name__ == "__main__":
    streamer_mode = False
    if len(sys.argv) > 1:
        if sys.argv[1] == 'STREAMER':
            streamer_mode = True
    match = Match(
        game_speed=100,
        self_play=True,
        team_size=1,
        state_setter=AugmentSetter(WallDribble(),
                                   shuffle_within_teams=True,
                                   swap_front_back=False,
                                   swap_teams=False
                                   ),
        obs_builder=ExpandAdvancedObs(),
        # action_parser=KBMAction(n_bins=N_BINS),
        action_parser=DiscreteAction(),
        terminal_conditions=[TimeoutCondition(round(10 // T_STEP)),
                             GoalScoredCondition()],
        reward_function=anneal_rewards_fn(),
    )

    r = Redis(host="127.0.0.1", username="user1", password=os.environ["redis_user1_key"])
    RedisRolloutWorker(r, "ABADv1", match, past_version_prob=0, streamer_mode=streamer_mode).run()
