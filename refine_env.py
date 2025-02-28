import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------ 环境相关 ------
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, FlattenObservation

import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

# 多目标算法
from morl_baselines.multi_policy.envelope.envelope import Envelope

from gymnasium.envs.registration import registry

# if "microbio-v0" in registry:
#     del registry["microbio-v0"]
# if "mo-microbio-v0" in registry:
#     del registry["mo-microbio-v0"]


from gymnasium.envs.registration import register

# register(
#     id="microbio-v0",
#     entry_point="mo_gymnasium.envs.microbio.microbio:SingleEvolvingEnv",
#     nondeterministic=True,
# )
# register(
#     id="mo-microbio-v0",
#     entry_point="mo_gymnasium.envs.mo_microbio.mo_microbio:MOEvolvingEnv",
#     nondeterministic=True,
# )


def make_microbio_env():
    """
    标准 Gym 环境 microbio-v0
    并使用 RecordEpisodeStatistics 来记录信息。
    """
    env = gym.make("microbio-v0")
    env = FlattenObservation(env)
    env = RecordEpisodeStatistics(env)
    return env


def make_mo_microbio_env():
    """
    mo-microbio-v0 多目标 Gym 环境
    并使用 MORecordEpisodeStatistics 来记录信息。
    """
    env = mo_gym.make("mo-microbio-v0")
    env = FlattenObservation(env)
    env = MORecordEpisodeStatistics(env, gamma=0.98)
    return env
    
env1=make_microbio_env()
env2=make_mo_microbio_env()