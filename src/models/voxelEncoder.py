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
OUTPUT_DIM = 128  # encoder output size used by actor, critic, and OM head


class VoxelEncoder(nn.Module):
    """
    Encodes a flat observation vector (151-dim) into a fixed embedding.
    Splits the flat obs back into: voxel grid (125), opponent features (22), self stats (4).
    Voxel path:   embed block IDs -> 3D CNN -> pool -> project
    Opponent path: linear embed -> attention pool
    Stats path:   linear
    All three fused -> OUTPUT_DIM = 128
    """

    def __init__(self):
        super().__init__()

        self.blockEmbed = nn.Embedding(NUM_BLOCK_TYPES, 8)

        self.cnn = nn.Sequential(
            nn.Conv3d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, CNN_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),  # (B, 64, 2, 2, 2)
        )
        cnnFlatDim = CNN_DIM * 2 * 2 * 2  # 512

        self.cnnProj = nn.Linear(cnnFlatDim, CNN_DIM)  # (B, 64)

        # Attention over opponent feature vectors
        self.oppEmbed  = nn.Linear(OPP_DIM, ENTITY_DIM)    # (B, nOpp, 32)
        self.queryProj = nn.Linear(CNN_DIM, ENTITY_DIM)     # for attention query
        self.scale     = ENTITY_DIM ** -0.5

        self.statsEncoder = nn.Linear(SELF_DIM, STATS_DIM)  # (B, 16)

        fusedDim = CNN_DIM + ENTITY_DIM + STATS_DIM  # 112
        self.outputProj = nn.Sequential(
            nn.Linear(fusedDim, OUTPUT_DIM),
            nn.ReLU(),
        )

    def forward(self, flatObs: torch.Tensor) -> torch.Tensor:
        """
        flatObs: (B, OBS_DIM=151)
        returns: (B, OUTPUT_DIM=128)
        """
        B = flatObs.size(0)

        # Split flat obs back into components
        stats     = flatObs[:, :SELF_DIM]                           # (B, 4)
        oppFeats  = flatObs[:, SELF_DIM: SELF_DIM + MAX_OPPONENTS * OPP_DIM]  # (B, 22)
        voxelFlat = flatObs[:, SELF_DIM + MAX_OPPONENTS * OPP_DIM:]            # (B, 125)

        # --- Voxel CNN path ---
        # De-normalise back to int IDs for embedding
        voxelIds = (voxelFlat * (NUM_BLOCK_TYPES - 1)).long().clamp(0, NUM_BLOCK_TYPES - 1)
        gridEmb  = self.blockEmbed(voxelIds)  # (B, 125, 8)
        gridEmb  = gridEmb.view(B, VOXEL_GRID_DIM, VOXEL_GRID_DIM, VOXEL_GRID_DIM, 8)
        gridEmb  = gridEmb.permute(0, 4, 1, 2, 3)          # (B, 8, 5, 5, 5)
        cnnOut   = self.cnn(gridEmb).reshape(B, -1)          # (B, 512)
        cnnFeats = F.relu(self.cnnProj(cnnOut))              # (B, 64)

        # --- Opponent attention path ---
        oppFeats  = oppFeats.view(B, MAX_OPPONENTS, OPP_DIM)  # (B, 2, 11)
        oppEmb    = F.relu(self.oppEmbed(oppFeats))            # (B, 2, 32)
        query     = self.queryProj(cnnFeats).unsqueeze(2)      # (B, 32, 1)
        scores    = torch.bmm(oppEmb, query).squeeze(2) * self.scale  # (B, 2)
        attnW     = F.softmax(scores, dim=-1)                  # (B, 2)
        entityOut = torch.bmm(attnW.unsqueeze(1), oppEmb).squeeze(1)  # (B, 32)

        # --- Stats path ---
        statsOut = F.relu(self.statsEncoder(stats))  # (B, 16)

        fused = torch.cat([cnnFeats, entityOut, statsOut], dim=-1)  # (B, 112)
        return self.outputProj(fused)                                # (B, 128)