from __future__ import annotations
import numpy as np

# Action space
N_MOVE   = 3
N_TURN   = 3
N_ATTACK = 2
ACTION_ONEHOT_DIM = N_MOVE + N_TURN + N_ATTACK  # 8

NUM_AGENTS       = 3
PREDATOR_INDICES = [0, 1]
PREY_INDICES     = [2]

# 5x5x5 voxel grid per opponent (125 ints)
VOXEL_GRID_SIZE = 5
VOXEL_DIM       = VOXEL_GRID_SIZE ** 3  # 125

# Self features: pos_x, pos_z, yaw, life = 4
# Opponent features per opponent: rel_pos(2) + life(1) + last_action_oh(8) = 11
# Voxel grid (around self): 125
# Total: 4 + 2*11 + 125 = 151
MAX_OPPONENTS = 2
SELF_DIM      = 4
OPP_DIM       = 2 + 1 + ACTION_ONEHOT_DIM  # 11
OBS_DIM       = SELF_DIM + MAX_OPPONENTS * OPP_DIM + VOXEL_DIM  # 151

# Global state: all agents' obs concatenated = 3 * 151 = 453
GLOBAL_STATE_DIM = NUM_AGENTS * OBS_DIM

ARENA_CENTER = 10.0
ARENA_HALF   = 9.0

BLOCK_TO_ID = {
    'air': 0, 'stone': 1, 'stonebrick': 2, 'grass': 3,
    'dirt': 4, 'cobblestone': 5, 'sand': 6, 'gravel': 7,
}
DEFAULT_BLOCK_ID = 15
NUM_BLOCK_TYPES  = 16


def _actionToOnehot(moveIdx: int, turnIdx: int, attackIdx: int) -> np.ndarray:
    oh = np.zeros(ACTION_ONEHOT_DIM, dtype=np.float32)
    oh[moveIdx]                     = 1.0
    oh[N_MOVE + turnIdx]            = 1.0
    oh[N_MOVE + N_TURN + attackIdx] = 1.0
    return oh


def _opponentIndices(agentIdx: int) -> list[int]:
    if agentIdx in PREDATOR_INDICES:
        otherPred = [i for i in PREDATOR_INDICES if i != agentIdx]
        return otherPred + PREY_INDICES
    else:
        return PREDATOR_INDICES


def parseVoxelGrid(rawGrid: list) -> np.ndarray:
    """Convert raw block-name list from Malmo into int ID array of shape (125,)."""
    return np.array(
        [BLOCK_TO_ID.get(b, DEFAULT_BLOCK_ID) for b in rawGrid],
        dtype=np.int32,
    )


def flattenObs(
    agentIdx: int,
    obs: dict,
    obsAll: list[dict],
    lastActionsAll: list[tuple[int, int, int]],
) -> np.ndarray:
    """
    Flatten one agent's observation to shape (OBS_DIM=151,).
    Voxel grid is the agent's own 5x5x5 neighbourhood (from Malmo grid obs).
    """
    selfVec = np.array([
        (obs["pos"][0] - ARENA_CENTER) / ARENA_HALF,
        (obs["pos"][1] - ARENA_CENTER) / ARENA_HALF,
        obs["yaw"] / 180.0,
        obs["life"] / 20.0,
    ], dtype=np.float32)

    oppVecs = []
    for oppIdx in _opponentIndices(agentIdx):
        oppObs  = obsAll[oppIdx]
        relPos  = (oppObs["pos"] - obs["pos"]) / 20.0
        life    = np.array([oppObs["life"] / 20.0], dtype=np.float32)
        m, t, a = lastActionsAll[oppIdx]
        actionOh = _actionToOnehot(m, t, a)
        oppVecs.append(np.concatenate([relPos, life, actionOh]))  # (11,)

    # Agent's own voxel grid (5x5x5 = 125), normalised to [0,1]
    voxel  = obs["voxelGrid"].astype(np.float32) / (NUM_BLOCK_TYPES - 1)

    result = np.concatenate([selfVec] + oppVecs + [voxel])
    assert result.shape == (OBS_DIM,), f"obs dim mismatch: {result.shape}"
    return result


def flattenObsAll(
    obsAll: list[dict],
    lastActionsAll: list[tuple[int, int, int]],
) -> np.ndarray:
    """Returns (NUM_AGENTS, OBS_DIM)."""
    return np.stack([
        flattenObs(i, obsAll[i], obsAll, lastActionsAll)
        for i in range(NUM_AGENTS)
    ])


def buildGlobalState(flatObsAll: np.ndarray) -> np.ndarray:
    """(NUM_AGENTS, OBS_DIM) -> (GLOBAL_STATE_DIM,) = (453,)"""
    return flatObsAll.flatten()


def actionToOnehot(moveIdx: int, turnIdx: int, attackIdx: int) -> np.ndarray:
    return _actionToOnehot(moveIdx, turnIdx, attackIdx)