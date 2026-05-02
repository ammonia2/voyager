"""
prioritizedReplay.py
====================
Thread-safe Prioritized Experience Replay (PER) buffer for MASAC.

Stores JOINT transitions — one push per environment step (not per agent):
  obs0 / obs1       : (OBS_DIM,) float32  predator 0 / 1 observations
  preyObs           : (OBS_DIM,) float32  prey observation (for global state)
  act0 / act1       : (3,)       float32  [move_idx, turn_cont, attack_idx]
  rew0 / rew1       : scalar     float32  per-predator rewards
  nextObs0 / 1      : (OBS_DIM,) float32  next predator observations
  nextPreyObs       : (OBS_DIM,) float32  next prey observation
  done              : scalar     float32  (0 or 1)
  preyMove          : scalar     int64    ground-truth prey move (OM loss)
  preyTurn          : scalar     int64    ground-truth prey turn (OM loss)

References
----------
Schaul et al. (2015) "Prioritized Experience Replay"
Horgan et al. (2018) "Distributed Prioritized Experience Replay (Ape-X)"
"""
from __future__ import annotations
import threading
import numpy as np
from src.utils.obsUtils import OBS_DIM

ACTION_DIM = 3   # (move_idx, turn_cont, attack_idx)


# ---------------------------------------------------------------------------
# Sum-Tree
# ---------------------------------------------------------------------------

class _SumTree:
    """Binary sum-tree for O(log N) priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self._ptr     = 0

    def _leafIdx(self, dataIdx: int) -> int:
        return dataIdx + self.capacity - 1

    def update(self, dataIdx: int, priority: float):
        idx   = self._leafIdx(dataIdx)
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def add(self, priority: float) -> int:
        dataIdx   = self._ptr
        self.update(dataIdx, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        return dataIdx

    def sample(self, value: float) -> int:
        idx = 0
        while idx < self.capacity - 1:
            left  = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx    = right
        return idx - (self.capacity - 1)

    @property
    def totalPriority(self) -> float:
        return float(self.tree[0])

    def __len__(self) -> int:
        return self.capacity


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer  —  joint transitions
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Thread-safe Prioritized Experience Replay buffer (joint transitions).

    Parameters
    ----------
    capacity  : max number of transitions
    alpha     : priority exponent  (0=uniform, 1=full PER)
    beta      : IS-weight exponent (annealed to 1 over betaSteps)
    betaSteps : annealing duration
    epsilon   : priority floor to prevent zero priorities
    """

    def __init__(
        self,
        capacity:  int   = 200_000,
        alpha:     float = 0.6,
        beta:      float = 0.4,
        betaSteps: int   = 100_000,
        epsilon:   float = 1e-6,
    ):
        self.capacity  = capacity
        self.alpha     = alpha
        self.beta0     = beta
        self.betaSteps = betaSteps
        self.epsilon   = epsilon
        self._step     = 0
        self._size     = 0

        self._tree = _SumTree(capacity)
        self._lock = threading.Lock()

        # Pre-allocated storage — joint transitions
        self._obs0      = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self._obs1      = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self._preyObs   = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self._act0      = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self._act1      = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self._rew0      = np.zeros(capacity, dtype=np.float32)
        self._rew1      = np.zeros(capacity, dtype=np.float32)
        self._nObs0     = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self._nObs1     = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self._nPreyObs  = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self._dones     = np.zeros(capacity, dtype=np.float32)
        self._preyMove  = np.zeros(capacity, dtype=np.int64)
        self._preyTurn  = np.zeros(capacity, dtype=np.int64)

        self._maxPriority = 1.0

    def _currentBeta(self) -> float:
        frac = min(1.0, self._step / max(self.betaSteps, 1))
        return self.beta0 + frac * (1.0 - self.beta0)

    def push(
        self,
        obs0:        np.ndarray,   # (OBS_DIM,)
        obs1:        np.ndarray,   # (OBS_DIM,)
        preyObs:     np.ndarray,   # (OBS_DIM,)
        act0:        np.ndarray,   # (3,)
        act1:        np.ndarray,   # (3,)
        rew0:        float,
        rew1:        float,
        nextObs0:    np.ndarray,   # (OBS_DIM,)
        nextObs1:    np.ndarray,   # (OBS_DIM,)
        nextPreyObs: np.ndarray,   # (OBS_DIM,)
        done:        float,
        preyMove:    int,
        preyTurn:    int,
    ):
        """Add one joint environment-step transition. Thread-safe."""
        priority = self._maxPriority ** self.alpha
        with self._lock:
            idx = self._tree.add(priority)
            self._obs0[idx]      = obs0
            self._obs1[idx]      = obs1
            self._preyObs[idx]   = preyObs
            self._act0[idx]      = act0
            self._act1[idx]      = act1
            self._rew0[idx]      = float(rew0)
            self._rew1[idx]      = float(rew1)
            self._nObs0[idx]     = nextObs0
            self._nObs1[idx]     = nextObs1
            self._nPreyObs[idx]  = nextPreyObs
            self._dones[idx]     = float(done)
            self._preyMove[idx]  = int(preyMove)
            self._preyTurn[idx]  = int(preyTurn)
            self._size           = min(self._size + 1, self.capacity)
            self._step          += 1

    def sample(self, batchSize: int) -> dict:
        """
        Sample batchSize joint transitions with IS weights.

        Returns a dict with numpy arrays:
            obs0, obs1, preyObs, act0, act1,
            rew0, rew1, nextObs0, nextObs1, nextPreyObs,
            dones, preyMove, preyTurn, weights, dataIndices
        """
        with self._lock:
            total      = self._tree.totalPriority
            beta       = self._currentBeta()
            n          = self._size
            segLen     = total / batchSize
            dataIdxs   = np.empty(batchSize, dtype=np.int64)
            priorities = np.empty(batchSize, dtype=np.float64)

            for i in range(batchSize):
                lo  = i * segLen
                hi  = lo + segLen
                val = np.random.uniform(lo, hi)
                idx = self._tree.sample(val)
                dataIdxs[i]   = idx
                priorities[i] = self._tree.tree[self._tree._leafIdx(idx)]

            probMin = (priorities / total).min()
            maxW    = (probMin * n) ** (-beta)
            probs   = priorities / total
            weights = ((probs * n) ** (-beta) / maxW).astype(np.float32)

            return {
                "obs0":        self._obs0[dataIdxs].copy(),
                "obs1":        self._obs1[dataIdxs].copy(),
                "preyObs":     self._preyObs[dataIdxs].copy(),
                "act0":        self._act0[dataIdxs].copy(),
                "act1":        self._act1[dataIdxs].copy(),
                "rew0":        self._rew0[dataIdxs].copy(),
                "rew1":        self._rew1[dataIdxs].copy(),
                "nextObs0":    self._nObs0[dataIdxs].copy(),
                "nextObs1":    self._nObs1[dataIdxs].copy(),
                "nextPreyObs": self._nPreyObs[dataIdxs].copy(),
                "dones":       self._dones[dataIdxs].copy(),
                "preyMove":    self._preyMove[dataIdxs].copy(),
                "preyTurn":    self._preyTurn[dataIdxs].copy(),
                "weights":     weights,
                "dataIndices": dataIdxs,
            }

    def updatePriorities(self, dataIndices: np.ndarray, tdErrors: np.ndarray):
        """Update priorities from new TD errors. Thread-safe."""
        with self._lock:
            for idx, err in zip(dataIndices, tdErrors):
                priority = (abs(float(err)) + self.epsilon) ** self.alpha
                self._tree.update(int(idx), priority)
                if priority > self._maxPriority:
                    self._maxPriority = priority

    def __len__(self) -> int:
        return self._size

    @property
    def readyToSample(self) -> bool:
        """True once there are enough transitions to form one batch."""
        return self._size >= 256
