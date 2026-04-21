from __future__ import annotations
import MalmoPython
import json
from src.envs.mappoEnv import MalmoEnv as _MAPPOEnv
from src.envs.mappoEnv import MOVE_CMDS, PREY_TURN_CMDS as TURN_CMDS, ATTACK_CMDS, NUM_AGENTS, PREDATOR_INDICES, PREY_INDICES

MalmoEnv = _MAPPOEnv
