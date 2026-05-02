"""
masac/voxelEncoder.py
=====================
Exact mirror of src/models/voxelEncoder.py, scoped to the MASAC model
package so the MASAC stack is fully self-contained.

Architecture
------------
  obs (147-dim) → split → [voxel 5×5×5 | opp feats | self stats]
  Voxel path  : Embedding(16,8) → 3D-CNN → AdaptivePool(2×2×2) → Linear → 64-d
  Opp path    : Linear(9,32) per opp → attention pool → 32-d
  Stats path  : Linear(4,16) → 16-d
  Fuse (112-d) → Linear → ReLU → OUTPUT_DIM = 128

All constants are re-imported from obsUtils to stay in sync with the rest
of the codebase (no magic numbers duplicated here).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import (
    VOXEL_GRID_SIZE as VOXEL_GRID_DIM,
    NUM_BLOCK_TYPES,
    SELF_DIM,
    OPP_DIM,
    MAX_OPPONENTS,
    OBS_DIM,
)

CNN_DIM    = 64
ENTITY_DIM = 32
STATS_DIM  = 16
OUTPUT_DIM = 128   # downstream models depend on this constant


class VoxelEncoder(nn.Module):
    """
    Encodes a flat (OBS_DIM=147) observation vector into a 128-d embedding.

    Observation layout (see obsUtils.py):
      [0   : 4 )   self stats  (pos_x, pos_z, yaw, life)
      [4   : 22)   opp feats   (2 opps × 9 = 18  float)
      [22  :147)   voxel grid  (5×5×5 = 125 int, normalised to [0,1])
    """

    def __init__(self):
        super().__init__()

        # Voxel block embedding
        self.blockEmbed = nn.Embedding(NUM_BLOCK_TYPES, 8)

        # 3-D CNN over the 5×5×5 embedded grid
        self.cnn = nn.Sequential(
            nn.Conv3d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, CNN_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),   # → (B, 64, 2, 2, 2)
        )
        cnnFlatDim = CNN_DIM * 2 * 2 * 2     # 512
        self.cnnProj = nn.Linear(cnnFlatDim, CNN_DIM)

        # Attention-pooled opponent embeddings
        self.oppEmbed  = nn.Linear(OPP_DIM, ENTITY_DIM)    # (B, nOpp, 32)
        self.queryProj = nn.Linear(CNN_DIM, ENTITY_DIM)
        self.scale     = ENTITY_DIM ** -0.5

        # Self-stats encoder
        self.statsEncoder = nn.Linear(SELF_DIM, STATS_DIM)

        fusedDim = CNN_DIM + ENTITY_DIM + STATS_DIM   # 112
        self.outputProj = nn.Sequential(
            nn.Linear(fusedDim, OUTPUT_DIM),
            nn.ReLU(),
        )

    def forward(self, flatObs: torch.Tensor) -> torch.Tensor:
        """
        flatObs : (B, OBS_DIM=147)
        returns : (B, OUTPUT_DIM=128)
        """
        B = flatObs.size(0)

        # Split
        stats     = flatObs[:, :SELF_DIM]                                        # (B, 4)
        oppFeats  = flatObs[:, SELF_DIM: SELF_DIM + MAX_OPPONENTS * OPP_DIM]    # (B, 18)
        voxelFlat = flatObs[:, SELF_DIM + MAX_OPPONENTS * OPP_DIM:]              # (B, 125)

        # --- Voxel CNN ---
        voxelIds = (voxelFlat * (NUM_BLOCK_TYPES - 1)).long().clamp(0, NUM_BLOCK_TYPES - 1)
        gridEmb  = self.blockEmbed(voxelIds)                      # (B, 125, 8)
        gridEmb  = gridEmb.view(B, VOXEL_GRID_DIM, VOXEL_GRID_DIM, VOXEL_GRID_DIM, 8)
        gridEmb  = gridEmb.permute(0, 4, 1, 2, 3)                # (B, 8, 5, 5, 5)
        cnnOut   = self.cnn(gridEmb).reshape(B, -1)               # (B, 512)
        cnnFeats = F.relu(self.cnnProj(cnnOut))                   # (B, 64)

        # --- Opponent attention ---
        oppFeats = oppFeats.view(B, MAX_OPPONENTS, OPP_DIM)       # (B, 2, 9)
        oppEmb   = F.relu(self.oppEmbed(oppFeats))                # (B, 2, 32)
        query    = self.queryProj(cnnFeats).unsqueeze(2)          # (B, 32, 1)
        scores   = torch.bmm(oppEmb, query).squeeze(2) * self.scale  # (B, 2)
        attnW    = F.softmax(scores, dim=-1)
        entityOut = torch.bmm(attnW.unsqueeze(1), oppEmb).squeeze(1)  # (B, 32)

        # --- Stats ---
        statsOut = F.relu(self.statsEncoder(stats))               # (B, 16)

        fused = torch.cat([cnnFeats, entityOut, statsOut], dim=-1)  # (B, 112)
        return self.outputProj(fused)                               # (B, 128)
