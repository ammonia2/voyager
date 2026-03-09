from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_SIZE    = 7
ENTITY_FEATS = 5  # (x, z, yaw, life, isSameTeam)

# entities are what the agent observes
class EntityAttention(nn.Module):
    """Attention pooling over variable-length nearby entity list."""
    def __init__(self, entityDim: int, hiddenDim: int):
        super().__init__()
        self.entityEmbed = nn.Linear(ENTITY_FEATS, entityDim) # embedding of cnn features
        self.queryProj   = nn.Linear(hiddenDim, entityDim)
        self.scale       = entityDim ** -0.5

    def forward(self, cnnFeats: torch.Tensor, entities: torch.Tensor, entityMask: torch.Tensor) -> torch.Tensor:
        """
        cnnFeats:   (B, hiddenDim)
        entities:   (B, maxEntities, ENTITY_FEATS)
        entityMask: (B, maxEntities) — 1 for real, 0 for padding
        returns:    (B, entityDim)
        """
        # B: number of batches being processed
        entityEmb = F.relu(self.entityEmbed(entities))          # (B, E, entityDim)
        query     = self.queryProj(cnnFeats).unsqueeze(2)        # (B, entityDim, 1)
        scores    = torch.bmm(entityEmb, query).squeeze(2)      # (B, E)
        scores    = scores * self.scale
        scores    = scores.masked_fill(entityMask == 0, -1e9)
        attnW     = F.softmax(scores, dim=-1)                   # (B, E)
        out       = torch.bmm(attnW.unsqueeze(1), entityEmb)    # (B, 1, entityDim)
        return out.squeeze(1)                                   # (B, entityDim)


class VoxelEncoder(nn.Module):
    """
    Encodes agent observation into a fixed-size feature vector.
    Input:  7x7 voxel grid + nearby entity list + agent stats
    Output: feature vector of size outputDim
    """
    def __init__(self, numBlockTypes: int = 16, cnnDim: int = 64, entityDim: int = 32, outputDim: int = 128):
        super().__init__()

        # Embed block type IDs before CNN
        self.blockEmbed = nn.Embedding(numBlockTypes, 8)

        # CNN over 7x7 spatial grid
        self.cnn = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),   # (B, 32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, cnnDim, kernel_size=3, padding=1),  # (B, 64, 7, 7)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),                  # (B, 64, 3, 3)
        )
        cnnFlatDim = cnnDim * 3 * 3  # 576

        # Project CNN output to hiddenDim before attention query
        self.cnnProj = nn.Linear(cnnFlatDim, cnnDim)

        # Attention over nearby entities (returns combined 
        # embeddings of each entity weighted by attention score)
        self.entityAttn = EntityAttention(entityDim, cnnDim)

        # Agent stats encoder (pos x, pos z, yaw, life)
        self.statsEncoder = nn.Linear(4, 16)

        # Final projection
        fusedDim = cnnDim + entityDim + 16
        self.outputProj = nn.Sequential(
            nn.Linear(fusedDim, outputDim),
            nn.ReLU(),
        )

    def forward(self, voxelGrid: torch.Tensor, entities: torch.Tensor,
                entityMask: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """
        voxelGrid:  (B, 49)     int block type IDs
        entities:   (B, E, 5)   nearby entity features
        entityMask: (B, E)      1=real entity, 0=padding
        stats:      (B, 4)      [x, z, yaw, life]
        returns:    (B, outputDim)
        """
        B = voxelGrid.size(0)

        # Embed + reshape to spatial grid
        gridEmb  = self.blockEmbed(voxelGrid.long())       # (B, 49, 8)
        gridEmb  = gridEmb.view(B, GRID_SIZE, GRID_SIZE, 8).permute(0, 3, 1, 2)  # (B, 8, 7, 7)

        # CNN
        cnnOut   = self.cnn(gridEmb).reshape(B, -1)           # (B, 576)
        cnnFeats = F.relu(self.cnnProj(cnnOut))            # (B, cnnDim)

        # Entity attention
        entityFeats = self.entityAttn(cnnFeats, entities, entityMask)  # (B, entityDim)

        # Stats
        statsFeats = F.relu(self.statsEncoder(stats))      # (B, 16)

        # Fuse all
        fused = torch.cat([cnnFeats, entityFeats, statsFeats], dim=-1)
        return self.outputProj(fused)                      # (B, outputDim)