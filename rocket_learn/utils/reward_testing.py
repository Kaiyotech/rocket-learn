import rlgym
from CoyoteBot.mybots_utils.mybots_rewards import *
from CoyoteBot.mybots_utils.mybots_statesets import *
from CoyoteBot.mybots_utils.mybots_terminals import *
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards


def anneal_rewards_fn():  # TODO this is throwing an error

    max_steps = 100_000_000  # TODO tune this some
    # when annealing, change the weights between 1 and 2, 2 is new
    reward1 = MyOldRewardFunction(
        team_spirit=0,
        goal_w=10,
        shot_w=5,
        save_w=5,
        demo_w=0,
        above_w=0,
        got_demoed_w=0,
        behind_ball_w=0,
        save_boost_w=0.03,
        concede_w=0,
        velocity_w=0,
        velocity_pb_w=0,
        velocity_bg_w=0.5,
        ball_touch_w=4,
    )
    reward2 = MyRewardFunction(
        team_spirit=0,
        goal_w=10,
        shot_w=5,
        save_w=5,
        demo_w=0,
        above_w=0,
        got_demoed_w=0,
        behind_ball_w=0,
        save_boost_w=0.03,
        concede_w=0,
        velocity_w=0,
        velocity_pb_w=0,
        velocity_bg_w=0.5,
        ball_touch_w=1,
    )

    alternating_rewards_steps = [reward1, max_steps, reward2]

    return AnnealRewards(*alternating_rewards_steps, mode=AnnealRewards.STEP)


env = rlgym.make(
        reward_fn=DoubleTapReward(),
        game_speed=1,
        state_setter=AugmentSetter(WallDribble(),
                                   shuffle_within_teams=True,
                                   swap_front_back=False,
                                   swap_left_right=False,
                                   swap_teams=False,
                                   ),
        terminal_conditions=[BallTouchGroundCondition()],
        self_play=True,
        )
try:
    while True:
        env.reset()
        done = False
        while not done:
            # Here we sample a random action. If you have an agent, you would get an action from it here.
            # action = env.action_space.sample()
            action = [1, 0, 0, 0, 0, 0, 0, 0] * 2

            next_obs, reward, done, gameinfo = env.step(action)

            if any(reward) > 0:
                print(reward)
                pass

            obs = next_obs

finally:
    env.close()
