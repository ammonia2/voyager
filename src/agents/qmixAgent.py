from __future__ import annotations
from agents.qmixMixer import Mixer
import torch, torch.nn.functional as F
import torch.nn as nn

N_MOVE   = 3
N_TURN   = 3
N_ATTACK = 2

class AgentQNetwork(nn.Module):
    """
    Deep Recurrrent Q-Network for agents
    GRU used for the recurrent part
    """
    def __init__(self, obsDim: int, hiddenDim: int = 64):
        super().__init__()
        self.gru = nn.GRU(obsDim, hiddenDim, batch_first=True)

        # separate Q-value heads per action dimension
        self.moveHead = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead = nn.Linear(hiddenDim, N_TURN)
        self.attackHead = nn.Linear(hiddenDim, N_ATTACK)

    def forward(self, obs: torch.Tensor, hiddenState: torch.Tensor = None):
        """
        obs:         (B, T, obsDim) or (B, obsDim) for single step
        hiddenState: (1, B, hiddenDim)
        returns:     move (B, 3), turn (B, 3), attack (B, 2), hidden (1, B, hiddenDim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        out, newHidden = self.gru(obs, hiddenState)         # (B, T, hiddenDim)
        out = out[:, -1, :]                                 # take last timestep (B, hiddenDim)

        return self.moveHead(out), self.turnHead(out), self.attackHead(out), newHidden

class QMIX:
    def __init__(self, nAgents: int, obsDim: int, stateDim: int, 
                 hiddenDim: int = 64, lr: float = 1e-3, device: str = "cpu"):
        self.nAgents = nAgents
        self.hiddenDim = hiddenDim
        self.device = torch.device(device)

        # live networks
        self.agentNet = AgentQNetwork(obsDim, hiddenDim).to(self.device)
        self.mixer = Mixer(nAgents, stateDim).to(self.device)

        # target networks
        self.targetAgentNet = AgentQNetwork(obsDim, hiddenDim).to(self.device)
        self.targetMixer = Mixer(nAgents, stateDim).to(self.device)
        self.targetAgentNet.load_state_dict(self.agentNet.state_dict())
        self.targetMixer.load_state_dict(self.mixer.state_dict())

        # targets don't accumulate gradients
        for p in self.targetAgentNet.parameters(): p.requires_grad = False
        for p in self.targetMixer.parameters(): p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            list(self.agentNet.parameters()) + list(self.mixer.parameters()),
            lr=lr
        )

        # hidden states per agent (reset each episode)
        self.hiddenStates = None
        self.targetHiddenStates = None

    def initHiddenStates(self, batchSize: int = 1):
        """hidden states reset utility to be called each episode"""
        self.hiddenStates = torch.zeros(1, batchSize * self.nAgents, self.hiddenDim, device=self.device)
        self.targetHiddenStates = torch.zeros(1, batchSize * self.nAgents, self.hiddenDim, device=self.device)

    def updateTargets(self):
        """hard copy live weights into target networks"""
        self.targetAgentNet.load_state_dict(self.agentNet.state_dict())
        self.targetMixer.load_state_dict(self.mixer.state_dict())