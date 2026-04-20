"""
Random rollout test — verifies env, obs pipeline, and encoder forward pass.
Run with: python experiments/testRandomRollout.py
Requires 4 Minecraft clients running on ports 10000-10003.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.envs.malmoEnv import MalmoEnv, NUM_AGENTS
from src.agents.randomAgent import RandomAgent
from src.models.voxelEncoder import VoxelEncoder
from src.models.omHead import OMHead

MISSION_XML  = "configs/missionPredatorPrey.xml"
MAX_STEPS    = 200
MAX_ENTITIES = 3  # max agents in observation range at once


def obsToTensors(obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a single agent obs dict to model-ready tensors with batch dim."""
    voxelGrid = torch.tensor(obs["voxelGrid"], dtype=torch.long).unsqueeze(0)   # (1, 49)
    stats     = torch.tensor(
        [obs["pos"][0], obs["pos"][1], obs["yaw"], obs["life"]], dtype=torch.float32
    ).unsqueeze(0)                                                                # (1, 4)

    # Pad entity list to MAX_ENTITIES
    entities  = np.zeros((MAX_ENTITIES, 5), dtype=np.float32)
    mask      = np.zeros(MAX_ENTITIES, dtype=np.float32)
    for i, e in enumerate(obs["nearbyEntities"][:MAX_ENTITIES]):
        entities[i] = [e.get("x", 0), e.get("z", 0), e.get("yaw", 0),
                       e.get("life", 20), float(e.get("name", "").startswith("Pred"))]
        mask[i] = 1.0

    entityTensor = torch.tensor(entities).unsqueeze(0)   # (1, 3, 5)
    maskTensor   = torch.tensor(mask).unsqueeze(0)       # (1, 3)
    return voxelGrid, entityTensor, maskTensor, stats


def main():
    env     = MalmoEnv(MISSION_XML)
    agents  = [RandomAgent() for _ in range(NUM_AGENTS)]
    encoder = VoxelEncoder()
    oppHead = OMHead()

    print("Resetting environment...")
    obsAll = env.reset()
    print("Episode started.\n")

    totalRewards = [0.0] * NUM_AGENTS

    for step in range(MAX_STEPS):
        # Random actions for all agents
        actions = [agent.act() for agent in agents]

        # Step env
        obsAll, rewards, dones = env.step(actions)

        # Accumulate rewards
        for i, r in enumerate(rewards):
            totalRewards[i] += r

        # Forward pass through encoder + opponent head for agent 0 as sanity check
        voxel, ents, mask, stats = obsToTensors(obsAll[0])
        with torch.no_grad():
            features  = encoder(voxel, ents, mask, stats)
            moveP, turnP, attackP = oppHead.predictProbs(features)

        print(f"Step {step + 1:03d} | rewards: {[round(r, 2) for r in rewards]} "
              f"| enc shape: {features.shape} | done: {any(dones)}")

        if any(dones):
            print("\nEpisode finished.")
            break

    print(f"\nTotal rewards: {[round(r, 2) for r in totalRewards]}")


if __name__ == "__main__":
    main()