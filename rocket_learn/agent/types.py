from typing import TypedDict, Dict, Optional

from rocket_learn.agent.pretrained_policy import HardcodedAgent


class PretrainedAgent(TypedDict):
    prob: float  # Probability agent appears in training
    eval: bool  # Whether or not to include in eval pools
    p_deterministic_training: float  # Probability of using deterministic in training
    key: str  # The key to be used for the redis hash set, should be unique


PretrainedAgents = Dict[HardcodedAgent, PretrainedAgent]
