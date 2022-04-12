import os
import wandb
import numpy
from typing import Any

import torch.jit
from torch.nn import Linear, Sequential, ReLU

from redis import Redis

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.gamestates import PlayerData, GameState
from rewards import anneal_rewards_fn
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import ExpandAdvancedObs, SplitLayer


from Constants import *


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    config = dict(
        gamma=1 - (T_STEP / TIME_HORIZON),
        gae_lambda=0.95,
        learning_rate_critic=1e-4,
        learning_rate_actor=1e-4,
        ent_coef=0.01,
        vf_coef=1.,
        target_steps=1_000_000,
        batch_size=200_000,
        minibatch_size=100_000,
        n_bins=3,
        n_epochs=25,
        iterations_per_save=10
    )
    run_id = "3825dsfe"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="wandb_store", name="ABADv1", project="ABAD", entity="kaiyotech", id=run_id, config=config)
    logger.name = "run2"

    redis = Redis(username="user1", password=os.environ["redis_user1_key"])

    # ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    def obs():
        return ExpandAdvancedObs()

    def rew():
        return anneal_rewards_fn()

    def act():
        return KBMAction()  # KBMAction(n_bins=N_BINS)

    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    rollout_gen = RedisRolloutGenerator(redis, obs, rew, act,
                                        logger=logger,
                                        save_every=logger.config.iterations_per_save)

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    split = (3, 3, 2, 2, 2)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 107

    critic = Sequential(
        Linear(state_dim, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, 1)
    )

    actor = DiscretePolicy(Sequential(
        Linear(state_dim, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, total_output),
        SplitLayer(splits=split)
    ), split)

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": logger.config.learning_rate_actor},
        {"params": critic.parameters(), "lr": logger.config.learning_rate_critic}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=logger.config.ent_coef,
        n_steps=logger.config.target_steps,  # target steps per rollout?
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.n_epochs,
        gamma=logger.config.gamma,
        gae_lambda=logger.config.gae_lambda,
        vf_coef=logger.config.vf_coef,
        logger=logger,
        device="cuda",
    )

    # SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="checkpoint_save_directory")
