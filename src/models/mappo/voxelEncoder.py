from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import (
    VOXEL_GRID_SIZE as VOXEL_GRID_DIM, NUM_BLOCK_TYPES, SELF_DIM, OPP_DIM, MAX_OPPONENTS, OBS_DIM
)

CNN_DIM    = 64
ENTITY_DIM = 32
STATS_DIM  = 16
OUTPUT_DIM = 128


class VoxelEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blockEmbed = nn.Embedding(NUM_BLOCK_TYPES, 8)

        self.cnn = nn.Sequential(
            nn.Conv3d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, CNN_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        cnnFlatDim = CNN_DIM * 2 * 2 * 2
        self.cnnProj = nn.Linear(cnnFlatDim, CNN_DIM)

        self.oppEmbed  = nn.Linear(OPP_DIM, ENTITY_DIM)
        self.queryProj = nn.Linear(CNN_DIM, ENTITY_DIM)
        self.scale     = ENTITY_DIM ** -0.5

        self.statsEncoder = nn.Linear(SELF_DIM, STATS_DIM)

        fusedDim = CNN_DIM + ENTITY_DIM + STATS_DIM
        self.outputProj = nn.Sequential(
            nn.Linear(fusedDim, OUTPUT_DIM),
            nn.ReLU(),
        )

    def forward(self, flatObs: torch.Tensor) -> torch.Tensor:
        B = flatObs.size(0)
        stats     = flatObs[:, :SELF_DIM]
        oppFeats  = flatObs[:, SELF_DIM: SELF_DIM + MAX_OPPONENTS * OPP_DIM]
        voxelFlat = flatObs[:, SELF_DIM + MAX_OPPONENTS * OPP_DIM:]

        voxelIds = (voxelFlat * (NUM_BLOCK_TYPES - 1)).long().clamp(0, NUM_BLOCK_TYPES - 1)
        gridEmb  = self.blockEmbed(voxelIds)
        gridEmb  = gridEmb.view(B, VOXEL_GRID_DIM, VOXEL_GRID_DIM, VOXEL_GRID_DIM, 8)
        gridEmb  = gridEmb.permute(0, 4, 1, 2, 3)
        cnnOut   = self.cnn(gridEmb).reshape(B, -1)
        cnnFeats = F.relu(self.cnnProj(cnnOut))

        oppFeats  = oppFeats.view(B, MAX_OPPONENTS, OPP_DIM)
        oppEmb    = F.relu(self.oppEmbed(oppFeats))
        query     = self.queryProj(cnnFeats).unsqueeze(2)
        scores    = torch.bmm(oppEmb, query).squeeze(2) * self.scale
        attnW     = F.softmax(scores, dim=-1)
        entityOut = torch.bmm(attnW.unsqueeze(1), oppEmb).squeeze(1)

        statsOut = F.relu(self.statsEncoder(stats))
        fused = torch.cat([cnnFeats, entityOut, statsOut], dim=-1)
        return self.outputProj(fused)
