from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Action space dimensions
# ---------------------------------------------------------------------------
N_MOVE        = 3   # discrete: move 1 / move -1 / move 0
N_TURN        = 3   # discrete bins kept for prey; predator turn is CONTINUOUS (see below)
N_ATTACK      = 2   # discrete: attack 1 / attack 0  (predator only)
N_PREY_TURN   = 3   # prey turn is still discrete (same 3 cmds)

# One-hot sizes used when encoding an opponent's last action inside observations:
#   Predator:  move_oh(3) + turn_continuous(1) + attack_oh(2) = 6
#   Prey:      move_oh(3) + turn_oh(3)                        = 6
#   Both are 6-dim so OPP_DIM is uniform.
PRED_ACTION_OH_DIM = N_MOVE + 1 + N_ATTACK   # 6  (1 for continuous turn value)
PREY_ACTION_OH_DIM = N_MOVE + N_PREY_TURN     # 6  (no attack)
ACTION_OH_DIM      = PRED_ACTION_OH_DIM        # 6  (uniform, used in OPP_DIM)

NUM_AGENTS       = 3
PREDATOR_INDICES = [0, 1]
PREY_INDICES     = [2]

# 5×5×5 voxel grid per agent (125 ints)
VOXEL_GRID_SIZE = 5
VOXEL_DIM       = VOXEL_GRID_SIZE ** 3  # 125

# Self features: pos_x, pos_z, yaw, life = 4
# Opponent features per opponent: rel_pos(2) + life(1) + action_oh(6) = 9
# Voxel grid (around self): 125
# Total: 4 + 2*9 + 125 = 147
MAX_OPPONENTS = 2
SELF_DIM      = 4
OPP_DIM       = 2 + 1 + ACTION_OH_DIM   # 9
OBS_DIM       = SELF_DIM + MAX_OPPONENTS * OPP_DIM + VOXEL_DIM  # 147

# Global state: all agents' obs concatenated = 3 × 147 = 441
GLOBAL_STATE_DIM = NUM_AGENTS * OBS_DIM

ARENA_CENTER = 10.0
ARENA_HALF   = 9.0

BLOCK_TO_ID = {
    'air': 0, 'stone': 1, 'stonebrick': 2, 'grass': 3,
    'dirt': 4, 'cobblestone': 5, 'sand': 6, 'gravel': 7,
}
DEFAULT_BLOCK_ID = 15
NUM_BLOCK_TYPES  = 16


# ---------------------------------------------------------------------------
# Action → observation one-hot helpers
# ---------------------------------------------------------------------------

def _predActionToOH(
    moveIdx: int, turnCont: float, attackIdx: int
) -> np.ndarray:
    """
    Encodes a predator's last (move, continuous-turn, attack) as a 6-dim OH.
      [0:3]  move one-hot
      [3]    continuous turn value in [-1, 1]
      [4:6]  attack one-hot
    """
    oh = np.zeros(PRED_ACTION_OH_DIM, dtype=np.float32)
    oh[moveIdx]       = 1.0
    oh[N_MOVE]        = float(np.clip(turnCont, -1.0, 1.0))  # index 3
    oh[N_MOVE + 1 + attackIdx] = 1.0                          # index 4 or 5
    return oh


def _preyActionToOH(moveIdx: int, turnIdx: int) -> np.ndarray:
    """
    Encodes a prey's last (move, discrete-turn) as a 6-dim OH.
      [0:3]  move one-hot
      [3:6]  turn one-hot
    (No attack — prey never punches.)
    """
    oh = np.zeros(PREY_ACTION_OH_DIM, dtype=np.float32)
    oh[moveIdx]           = 1.0
    oh[N_MOVE + turnIdx]  = 1.0
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


# ---------------------------------------------------------------------------
# Observation flattening
# ---------------------------------------------------------------------------

def flattenObs(
    agentIdx: int,
    obs: dict,
    obsAll: list[dict],
    lastActionsAll: list[tuple],
) -> np.ndarray:
    """
    Flatten one agent's observation to shape (OBS_DIM=147,).
    lastActionsAll stores mixed-type tuples:
      Predator: (move_idx: int, turn_cont: float, attack_idx: int)
      Prey:     (move_idx: int, turn_idx:  int,   0)
    """
    selfVec = np.array([
        (obs["pos"][0] - ARENA_CENTER) / ARENA_HALF,
        (obs["pos"][1] - ARENA_CENTER) / ARENA_HALF,
        obs["yaw"] / 180.0,
        obs["life"] / 20.0,
    ], dtype=np.float32)

    oppVecs = []
    for oppIdx in _opponentIndices(agentIdx):
        oppObs = obsAll[oppIdx]
        relPos = (oppObs["pos"] - obs["pos"]) / 20.0
        life   = np.array([oppObs["life"] / 20.0], dtype=np.float32)
        act    = lastActionsAll[oppIdx]
        if oppIdx in PREDATOR_INDICES:
            # act = (move_idx, turn_cont, attack_idx)
            actionOh = _predActionToOH(int(act[0]), float(act[1]), int(act[2]))
        else:
            # act = (move_idx, turn_idx, 0) — prey
            actionOh = _preyActionToOH(int(act[0]), int(act[1]))
        oppVecs.append(np.concatenate([relPos, life, actionOh]))  # (9,)

    # Agent's own voxel grid (5×5×5 = 125), normalised to [0, 1]
    voxel = obs["voxelGrid"].astype(np.float32) / (NUM_BLOCK_TYPES - 1)

    result = np.concatenate([selfVec] + oppVecs + [voxel])
    assert result.shape == (OBS_DIM,), f"obs dim mismatch: {result.shape}"
    return result


def flattenObsAll(
    obsAll: list[dict],
    lastActionsAll: list[tuple],
) -> np.ndarray:
    """Returns (NUM_AGENTS, OBS_DIM)."""
    return np.stack([
        flattenObs(i, obsAll[i], obsAll, lastActionsAll)
        for i in range(NUM_AGENTS)
    ])


def buildGlobalState(flatObsAll: np.ndarray) -> np.ndarray:
    """(NUM_AGENTS, OBS_DIM) -> (GLOBAL_STATE_DIM,) = (441,)"""
    return flatObsAll.flatten()


# Legacy alias kept for any callers that still use the old name
def actionToOnehot(moveIdx: int, turnIdx: int, attackIdx: int) -> np.ndarray:
    return _predActionToOH(moveIdx, float(turnIdx) / 1.0, attackIdx)