from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from trueskill import Rating
from redis import Redis

from rocket_learn.agent.types import PretrainedAgents


class BaseMatchmaker(ABC):
    @abstractmethod
    def generate_matchup(self, redis: Redis, n_agents: int, evaluate: bool) -> Tuple[List[Union[str, int]], List[Rating], bool]:
        """
        Function to compute the reward for a player. This function is given a player argument, and it is expected that
        the reward returned by this function will be for that player.

        :param redis: The Redis client that hosts the ratings database and latest model
        :param n_agents: The number of agents to appear in the match.
        :param evaluate: A boolean representing whether or not the matchup generated is for an evaluation match.

        :return: A tuple with 2 parallel lists and a bool. The first is a list of version names, the second is a list of ratings, the third is whether or not the match is an evaluation match. -1 in version name means latest.
        """
        raise NotImplementedError
