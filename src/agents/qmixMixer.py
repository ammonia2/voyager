from __future__ import annotations
import torch, torch.nn.functional as F
import torch.nn as nn

class Mixer(nn.Module):
    """
    Mixing network + hypernetworks as described in the QMIX paper.
    Hidden layer: 32 units, ELU activation.
    Hypernetworks: single hidden layer of 64 units, ReLU, abs output.
    """
    def __init__(self, nAgents: int, stateDim: int, mixingHidden: int = 32, hyperHidden: int = 64):
        super().__init__()
        self.nAgents = nAgents
        self.mixingHidden = mixingHidden

        # hypernetwork for W1
        self.hyperW1 = nn.Sequential(
            nn.Linear(stateDim, hyperHidden),
            nn.ReLU(),
            nn.Linear(hyperHidden, nAgents * mixingHidden)
        )

        # hypernetwork for W2
        self.hyperW2 = nn.Sequential(
            nn.Linear(stateDim, hyperHidden),
            nn.ReLU(),
            nn.Linear(hyperHidden, mixingHidden)
        )

        self.hyperB1 = nn.Linear(stateDim, mixingHidden)
        self.hyperB2 = nn.Sequential(
            nn.Linear(stateDim, mixingHidden),
            nn.ReLU(),
            nn.Linear(mixingHidden, 1)
        )

    def forward(self, agentQVals: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        agentQvals: (B, nAgents) — one Q-value per agent
        state:      (B, stateDim)
        returns:    (B, 1) — Q_tot
        """
        B = agentQVals.size(0)

        # reshaping to matrices to batch matmul
        w1 = torch.abs(self.hyperW1(state))
        w1 = w1.view(B, self.nAgents, self.mixingHidden)
        b1 = self.hyperB1(state).unsqueeze(1)

        # first layer of mixer
        x = agentQVals.unsqueeze(1)
        x = F.elu(torch.bmm(x, w1) +b1)
        
        w2 = torch.abs(self.hyperW2(state))
        w2 = w2.view(B, self.mixingHidden, 1)
        b2 = self.hyperB2(state).unsqueeze(1)

        # output layer of mixer
        qTot = torch.bmm(x, w2) + b2
        return qTot.squeeze(-1).squeeze(-1)