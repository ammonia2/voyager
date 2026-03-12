from __future__ import annotations
import numpy as np
import random
from dataclasses import dataclass, field

@dataclass
class Episode:
    observations: list = field(default_factory=list) # to prevent sharing same list object across instances
    actions: list = field(default_factory=list)
    rewards:list = field(default_factory=list)
    states: list = field(default_factory=list)
    dones:list = field(default_factory=list)

    def __len__(self):
        return len(self.rewards)

class ReplayBuffer:
    def __init__(self, maxEpisodes: int = 1000):
        self.maxEpisodes = maxEpisodes
        self.buffer: list[Episode] = []

    def addEpisode(self, episode: Episode):
        if len(self.buffer) >= self.maxEpisodes:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def sample(self, batchSize: int) -> list[Episode]:
        return random.sample(self.buffer, min(batchSize, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)
