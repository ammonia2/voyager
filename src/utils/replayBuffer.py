from __future__ import annotations
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Stores transitions for all agents simultaneously.
    Each transition: (obsAll, actionsAll, rewardsAll, nextObsAll, dones)
      obsAll:     (NUM_AGENTS, OBS_DIM) float32
            actionsAll: (NUM_AGENTS, 3)       float - (moveIdx, turnVal, attackIdx)
      rewardsAll: (NUM_AGENTS,)         float32
      nextObsAll: (NUM_AGENTS, OBS_DIM) float32
      dones:      (NUM_AGENTS,)         bool
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obsAll: np.ndarray,
        actionsAll: np.ndarray,
        rewardsAll: np.ndarray,
        nextObsAll: np.ndarray,
        dones: np.ndarray,
    ):
        self.buffer.append((
            obsAll.astype(np.float32),
            actionsAll.astype(np.float32),
            rewardsAll.astype(np.float32),
            nextObsAll.astype(np.float32),
            dones.astype(np.float32),
        ))

    def sample(self, batchSize: int) -> tuple:
        """
        Returns batched numpy arrays:
          obsAll:     (B, NUM_AGENTS, OBS_DIM)
          actionsAll: (B, NUM_AGENTS, 3)
          rewardsAll: (B, NUM_AGENTS)
          nextObsAll: (B, NUM_AGENTS, OBS_DIM)
          dones:      (B, NUM_AGENTS)
        """
        batch = random.sample(self.buffer, batchSize)
        obsAll, actionsAll, rewardsAll, nextObsAll, dones = zip(*batch)
        return (
            np.stack(obsAll),
            np.stack(actionsAll),
            np.stack(rewardsAll),
            np.stack(nextObsAll),
            np.stack(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)