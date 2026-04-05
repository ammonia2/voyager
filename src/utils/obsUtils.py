from __future__ import annotations
import numpy as np

GRID_SIZE         = 7
N_MOVE            = 3
N_TURN            = 3
N_ATTACK          = 2
ACTION_ONEHOT_DIM = N_MOVE + N_TURN + N_ATTACK  # 8

NUM_AGENTS        = 3
PREDATOR_INDICES  = [0, 1]
PREY_INDICES      = [2]

# Each agent sees 2 opponents regardless of role:
#   predators: 1 prey + 1 teammate predator
#   prey:      2 predators
# self(4) + 2*(rel_pos(2)+life(1)+action_oh(8)) + voxel(49) = 75
MAX_OPPONENTS = 2
OBS_DIM       = 4 + MAX_OPPONENTS * (2 + 1 + ACTION_ONEHOT_DIM) + GRID_SIZE * GRID_SIZE  # 75


def _actionToOnehot(moveIdx: int, turnIdx: int, attackIdx: int) -> np.ndarray:
    """Concatenated per-head one-hot of length 8."""
    oh = np.zeros(ACTION_ONEHOT_DIM, dtype=np.float32)
    oh[moveIdx]                   = 1.0
    oh[N_MOVE + turnIdx]          = 1.0
    oh[N_MOVE + N_TURN + attackIdx] = 1.0
    return oh


def _opponentIndices(agentIdx: int) -> list[int]:
    """
    Predators treat the other predator as a generic opponent alongside the prey.
    Prey treats both predators as opponents.
    """
    if agentIdx in PREDATOR_INDICES:
        otherPred = [i for i in PREDATOR_INDICES if i != agentIdx]
        return otherPred + PREY_INDICES   # [other_pred, prey] — always length 2
    else:
        return PREDATOR_INDICES           # [pred0, pred1]


ARENA_CENTER = 10.0   # arena runs 1-19, centre at 10
ARENA_HALF   = 9.0    # half-width for normalisation


def flattenObs(
    agentIdx: int,
    obs: dict,
    obsAll: list[dict],
    lastActionsAll: list[tuple[int, int, int]],
) -> np.ndarray:
    """
    Flatten one agent's observation to a fixed vector of shape (OBS_DIM=75,).
    agentIdx:       which agent (0-2)
    obs:            this agent's parsed obs dict from malmoEnv
    obsAll:         all 3 agents' parsed obs dicts
    lastActionsAll: list of (moveIdx, turnIdx, attackIdx) for all 3 agents

    Self position is encoded as arena-centre-relative (not absolute) so the
    policy generalises across random spawn locations instead of memorising
    spawn-specific behaviours.
    """
    selfVec = np.array([
        (obs["pos"][0] - ARENA_CENTER) / ARENA_HALF,   # x relative to centre
        (obs["pos"][1] - ARENA_CENTER) / ARENA_HALF,   # z relative to centre
        obs["yaw"] / 180.0,
        obs["life"] / 20.0,
    ], dtype=np.float32)

    oppVecs = []
    for oppIdx in _opponentIndices(agentIdx):
        oppObs = obsAll[oppIdx]
        relPos   = (oppObs["pos"] - obs["pos"]) / 20.0
        life     = np.array([oppObs["life"] / 20.0], dtype=np.float32)
        m, t, a  = lastActionsAll[oppIdx]
        actionOh = _actionToOnehot(m, t, a)
        oppVecs.append(np.concatenate([relPos, life, actionOh]))  # (11,)

    voxel  = obs["voxelGrid"].astype(np.float32) / 15.0
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
    """(NUM_AGENTS, OBS_DIM) -> (NUM_AGENTS * OBS_DIM,) = (225,)"""
    return flatObsAll.flatten()


def actionToOnehot(moveIdx: int, turnIdx: int, attackIdx: int) -> np.ndarray:
    return _actionToOnehot(moveIdx, turnIdx, attackIdx)