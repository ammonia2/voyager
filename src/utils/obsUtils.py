from __future__ import annotations
import numpy as np
import torch

MAX_ENTITIES = 3
ENTITY_FEATS = 5  # (x, z, yaw, life, isSameTeam)
OBS_DIM      = 49 + 2 + 1 + 1 + 1 + MAX_ENTITIES * ENTITY_FEATS  # 69

def flattenObs(obs: dict, agentId: int, agentNames: list[str]) -> np.ndarray:
    """
    Flattens raw obs dict from malmoEnv into a fixed-size vector.
    agentId:    0-3 integer, concatenated so shared network can differentiate agents
    agentNames: ordered list of agent name strings e.g. ['Predator1', ...]
    """
    voxel = obs["voxelGrid"]                          # (49,)
    pos   = obs["pos"]                                # (2,)
    life  = np.array([obs["life"] / 20.0])            # normalise to [0,1]
    yaw   = np.array([obs["yaw"] / 180.0])            # normalise to [-1,1]
    aid   = np.array([agentId / 3.0])                 # normalise to [0,1]

    # build entity matrix, zero-pad to MAX_ENTITIES
    entityArr = np.zeros((MAX_ENTITIES, ENTITY_FEATS), dtype=np.float32)
    for i, ent in enumerate(obs["nearbyEntities"][:MAX_ENTITIES]):
        isSameTeam = _isSameTeam(agentId, ent["name"], agentNames)
        entityArr[i] = [
            ent["x"] / 20.0,          # normalise to arena size
            ent["z"] / 20.0,
            ent.get("yaw", 0.0) / 180.0,
            ent.get("life", 20.0) / 20.0,
            float(isSameTeam),
        ]

    return np.concatenate([voxel, pos / 20.0, life, yaw, aid, entityArr.flatten()])

def _isSameTeam(agentId: int, entityName: str, agentNames: list[str]) -> bool:
    """Predators are indices 0,1 — Prey are 2,3."""
    predators = {agentNames[0], agentNames[1]}
    isPredator = agentId < 2
    entityIsPredator = entityName in predators
    return isPredator == entityIsPredator

def obsToTensor(obs: dict, agentId: int, agentNames: list[str], 
                device: torch.device) -> torch.Tensor:
    """Convenience wrapper — flat np array to tensor."""
    flat = flattenObs(obs, agentId, agentNames)
    return torch.tensor(flat, dtype=torch.float32, device=device)

def batchObsAllAgents(obsAll: list[dict], agentNames: list[str],
                      device: torch.device) -> torch.Tensor:
    """
    Converts obs for all agents into a single batched tensor.
    returns: (nAgents, obsDim)
    """
    return torch.stack([
        obsToTensor(obsAll[i], i, agentNames, device)
        for i in range(len(obsAll))
    ])