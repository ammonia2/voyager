from __future__ import annotations
import numpy as np

from src.utils.obsUtils import NUM_AGENTS, OBS_DIM, GLOBAL_STATE_DIM, PREDATOR_INDICES, PREY_INDICES


class RolloutBuffer:
    """
    Fixed-capacity on-policy rollout buffer for MAPPO.
    Stores one T-step rollout. No replay — reset after every update.

    Prey value estimates use a running mean reward as REINFORCE baseline
    (updated externally by the trainer). This avoids needing a separate
    per-agent critic for the prey.
    """

    def __init__(self, rolloutSteps: int):
        T = rolloutSteps
        self.T = T
        self.flatObsAll   = np.zeros((T, NUM_AGENTS, OBS_DIM),     dtype=np.float32)
        self.globalStates = np.zeros((T, GLOBAL_STATE_DIM),         dtype=np.float32)
        self.actions      = np.zeros((T, NUM_AGENTS, 3),            dtype=np.int64)
        self.rewards      = np.zeros((T, NUM_AGENTS),               dtype=np.float32)
        self.dones        = np.zeros(T,                             dtype=np.float32)
        self.logProbs     = np.zeros((T, len(PREDATOR_INDICES)),    dtype=np.float32)
        self.values       = np.zeros(T,                             dtype=np.float32)
        self.preyLogProbs = np.zeros(T,                             dtype=np.float32)
        # Prey values: filled with the running mean prey reward baseline
        self.preyValues   = np.zeros(T,                             dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        flatObsAll:   np.ndarray,   # (NUM_AGENTS, OBS_DIM)
        globalState:  np.ndarray,   # (GLOBAL_STATE_DIM,)
        actions:      np.ndarray,   # (NUM_AGENTS, 3)
        rewards:      np.ndarray,   # (NUM_AGENTS,)
        done:         bool,
        logProbs:     np.ndarray,   # (nPredators,)
        value:        float,        # centralised critic V(s) for predator GAE
        preyLogProb:  float,
        preyValue:    float,        # running mean baseline for prey REINFORCE
    ):
        t = self.ptr
        self.flatObsAll[t]   = flatObsAll
        self.globalStates[t] = globalState
        self.actions[t]      = actions
        self.rewards[t]      = rewards
        self.dones[t]        = float(done)
        self.logProbs[t]     = logProbs
        self.values[t]       = value
        self.preyLogProbs[t] = preyLogProb
        self.preyValues[t]   = preyValue
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.T

    def get(self, lastValue: float, preyLastValue: float) -> dict:
        """
        Returns the accumulated rollout as a dict of numpy arrays.
        lastValue:     GAE bootstrap for predator critic (0 if terminal)
        preyLastValue: GAE bootstrap for prey baseline (0 if terminal)
        """
        n = self.ptr
        return {
            "flatObsAll":    self.flatObsAll[:n],
            "globalStates":  self.globalStates[:n],
            "actions":       self.actions[:n],
            "rewards":       self.rewards[:n],
            "dones":         self.dones[:n],
            "logProbs":      self.logProbs[:n],
            "values":        self.values[:n],
            "lastValue":     lastValue,
            "preyLogProbs":  self.preyLogProbs[:n],
            "preyValues":    self.preyValues[:n],
            "preyLastValue": preyLastValue,
        }

    def reset(self):
        self.ptr = 0