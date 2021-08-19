import copy
import os
import pickle
import time
from typing import Generator, Iterator

import numpy as np
from redis import Redis
from torch import nn

from rlgym.envs import Match
from rlgym.gym import Gym
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator

# Hopefully we only need this one file, so this is where it belongs
from rocket_learn.simple_agents import PPOAgent
from rocket_learn.utils import util
from rocket_learn.utils.util import softmax

QUALITIES = "qualities"
MODEL_LATEST = "model-latest"
ROLLOUTS = "rollout"
VERSION_LATEST = "model-version"
OP_MODELS = "opponent_models"


class RedisRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, save_every=10):
        # **DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = Redis(host='127.0.0.1', port=6379, db=0)
        self.n_updates = 0
        self.save_every = save_every

        # TODO saving/loading
        for key in QUALITIES, MODEL_LATEST, ROLLOUTS, VERSION_LATEST, OP_MODELS:
            if self.redis.exists(key) > 0:
                self.redis.delete(key)

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            rollout = self.redis.lpop(ROLLOUTS)
            if rollout is not None:  # Assuming nil is converted to None by py-redis
                yield pickle.loads(rollout)
            else:
                time.sleep(1)  # Don't DOS Redis

    def _update_model(self, agent, version):  # TODO same as update_parameters?
        if self.redis.exists(MODEL_ACTOR_LATEST) > 0:
            self.redis.delete(MODEL_ACTOR_LATEST)
        if self.redis.exists(MODEL_CRITIC_LATEST) > 0:
            self.redis.delete(MODEL_CRITIC_LATEST)
        if self.redis.exists(VERSION_LATEST) > 0:
            self.redis.delete(VERSION_LATEST)

        actor_bytes = pickle.dumps(agent.actor.state_dict())
        critic_bytes = pickle.dumps(agent.critic.state_dict())

        self.redis.set(MODEL_ACTOR_LATEST, actor_bytes)
        self.redis.set(MODEL_CRITIC_LATEST, critic_bytes)
        self.redis.set(VERSION_LATEST, version)
        print("done setting")

    def _add_opponent(self, state_dict_dump):  # TODO use
        # Add to list
        self.redis.rpush(OP_MODELS, state_dict_dump)
        # Set quality
        qualities = [float(v) for v in self.redis.lrange(QUALITIES, 0, -1)]
        if qualities:
            quality = max(qualities)
        else:
            quality = 0.
        self.redis.rpush(QUALITIES, quality)

    def update_parameters(self, new_params):
        model_bytes = pickle.dumps(new_params)
        self.redis.set(MODEL_LATEST, model_bytes)
        self.redis.set(VERSION_LATEST, self.n_updates)
        if self.n_updates % self.save_every == 0:
            # self.redis.set(MODEL_N.format(self.n_updates // self.save_every), model_bytes)
            self._add_opponent(model_bytes)
        self.n_updates += 1


class RedisRolloutWorker:  # Provides RedisRolloutGenerator with rollouts via a Redis server
    def __init__(self, epic_rl_path, actor, critic, match: Match, current_version_prob=.8):
        # example pytorch stuff, delete later
        self.state_dim = 67
        self.action_dim = 8

        self.actor = actor
        self.critic = critic

        self.current_agent = PPOAgent(actor, critic)
        self.current_version_prob = current_version_prob

        # **DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = Redis(host='127.0.0.1', port=6379, db=0)
        self.match = match
        self.env = Gym(match=self.match, pipe_id=os.getpid(), path_to_rl=epic_rl_path, use_injector=True)
        self.n_agents = self.match.agents

    def _get_opponent_index(self):
        # Get qualities
        qualities = np.asarray([float(v) for v in self.redis.lrange(QUALITIES, 0, -1)])
        # Pick opponent
        probs = softmax(qualities)
        index = np.random.choice(len(probs), p=probs)
        return index, probs[index]

    def _update_opponent_quality(self, index, prob, rate):  # TODO use
        # Calculate delta
        n = self.redis.llen(QUALITIES)
        delta = rate / (n * prob)
        # lua script to read and update atomically
        self.redis.eval('''
            local q = tonumber(redis.call('LINDEX', KEYS[1], KEYS[2]))
            local delta = tonumber(ARGV[1])
            local new_q = q + delta
            return redis.call('LSET', KEYS[1], KEYS[2], new_q)
            ''', 2, QUALITIES, index, delta)

    def run(self):  # Mimics Thread
        n = 0
        while True:
            model_bytes = self.redis.get(MODEL_LATEST)
            latest_version = self.redis.get(VERSION_LATEST)
            if model_bytes is None:
                time.sleep(1)
                continue  # Wait for model to get published
            actor_dict, critic_dict = pickle.loads(model_bytes)
            latest_version = int(latest_version)

            updated_agent = PPOAgent(copy.deepcopy(self.actor), copy.deepcopy(self.critic))
            updated_agent.actor.load_state_dict(actor_dict)
            updated_agent.critic.load_state_dict(critic_dict)

            print(n, all(p1.data.ne(p2.data).sum() > 0 for p1, p2 in
                         zip(self.current_agent.actor.parameters(),
                             updated_agent.actor.parameters())))

            n += 1

            self.current_agent = updated_agent

            # TODO customizable past agent selection, should team only be same agent?
            agents = [(self.current_agent, latest_version, self.current_version_prob)]  # Use at least one current agent

            if self.n_agents > 1:
                # Ensure final proportion is same
                adjusted_prob = (self.current_version_prob * self.n_agents - 1) / (self.n_agents - 1)
                for i in range(self.n_agents - 1):
                    is_current = np.random.random() < adjusted_prob
                    if not is_current:
                        index, prob = self._get_opponent_index()
                        version = OP_MODELS
                        actor_dict, critic_dict = pickle.loads(self.redis.lindex(OP_MODELS, index))
                        selected_agent = PPOAgent(copy.deepcopy(self.actor), copy.deepcopy(self.critic))
                        selected_agent.actor.load_state_dict(actor_dict)
                        selected_agent.critic.load_state_dict(critic_dict)
                    else:
                        prob = self.current_version_prob
                        version = latest_version
                        selected_agent = self.current_agent

                    agents.append((selected_agent, version, prob))

            np.random.shuffle(agents)

            rollouts = util.generate_episode(self.env, [agent for agent, version, prob in agents])

            self.redis.rpush(ROLLOUTS, *(pickle.dumps(rollout) for rollout in rollouts))