from __future__ import annotations
import MalmoPython
import json
import time
import numpy as np
from pathlib import Path

# Action space per agent: [move, turn, attack]
# move:   0=forward, 1=backward, 2=stop
# turn:   0=left,    1=right,    2=none
# attack: 0=yes,     1=no
MOVE_CMDS   = ["move 1", "move -1", "move 0"]
TURN_CMDS   = ["turn -1", "turn 1", "turn 0"]
ATTACK_CMDS = ["attack 1", "attack 0"]

NUM_AGENTS   = 4
AGENT_NAMES  = ["Predator1", "Predator2", "Prey1", "Prey2"]
BASE_PORT    = 10000
GRID_SIZE    = 7 # could increase to 9x9 or decrease to 5x5
STEP_SLEEP   = 0.1  # seconds between steps

BLOCK_TO_ID = {
    'air': 0, 'stone': 1, 'stonebrick': 2, 'grass': 3,
    'dirt': 4, 'cobblestone': 5, 'sand': 6, 'gravel': 7,
}
DEFAULT_BLOCK_ID = 15  # unknown block type

class MalmoEnv:
    def __init__(self, missionXmlPath: str):
        self.missionXml = Path(missionXmlPath).read_text()
        self.agentHosts = [MalmoPython.AgentHost() for _ in range(NUM_AGENTS)]
        self.clientPool  = self._buildClientPool()
        self.prevHealth = []

    def _buildClientPool(self) -> MalmoPython.ClientPool:
        pool = MalmoPython.ClientPool()
        for i in range(NUM_AGENTS):
            pool.add(MalmoPython.ClientInfo("127.0.0.1", BASE_PORT + i))
        return pool

    def reset(self) -> list[dict]:
        mission       = MalmoPython.MissionSpec(self.missionXml, True)
        missionRecord = MalmoPython.MissionRecordSpec()
        experimentId  = str(int(time.time()))  # unique per episode

        for i, host in enumerate(self.agentHosts):
            host.startMission(mission, self.clientPool, missionRecord, i, experimentId)
            if i == 0:
                time.sleep(30)  # role 0 needs time to start the server
            else:
                time.sleep(1)

        self._waitForAllAgents()
        obsAll = self._getObsAll()
        self.prevHealth = [obs["life"] for obs in obsAll]
        return self._getObsAll()

    def step(self, actions: list[tuple[int, int, int]]) -> tuple[list, list, list]:
        """
        actions: list of (moveIdx, turnIdx, attackIdx) per agent
        returns: (obsAll, rewardsAll, donesAll)
        """
        for i, (host, action) in enumerate(zip(self.agentHosts, actions)):
            moveIdx, turnIdx, attackIdx = action
            host.sendCommand(MOVE_CMDS[moveIdx])
            host.sendCommand(TURN_CMDS[turnIdx])
            host.sendCommand(ATTACK_CMDS[attackIdx])

        time.sleep(STEP_SLEEP)

        obsAll     = self._getObsAll()
        rewardsAll = self._getRewardsAll(obsAll)
        donesAll   = self._getDonesAll()

        return obsAll, rewardsAll, donesAll

    def _waitForAllAgents(self):
        for host in self.agentHosts:
            worldState = host.getWorldState()
            while not worldState.has_mission_begun:
                time.sleep(0.1)
                worldState = host.getWorldState()

    def _getObsAll(self) -> list[dict]:
        obs = []
        for host in self.agentHosts:
            worldState = host.getWorldState()
            if worldState.number_of_observations_since_last_state > 0:
                raw = json.loads(worldState.observations[-1].text)
                obs.append(self._parseObs(raw))
            else:
                obs.append(self._emptyObs())
        return obs

    def _parseObs(self, raw: dict) -> dict:
        # Voxel grid: GRID_SIZE x GRID_SIZE flattened block types
        voxelGrid = raw.get("voxelObs", [])
        voxelArr = np.array(
            [BLOCK_TO_ID.get(b, DEFAULT_BLOCK_ID) for b in voxelGrid],
            dtype=np.float32
        )

        # Nearby entities: list of {name, x, y, z, life}
        nearbyEntities = raw.get("nearbyEntities", [])

        # Agent stats
        life = raw.get("Life", 20.0)
        xPos = raw.get("XPos", 0.0)
        zPos = raw.get("ZPos", 0.0)
        yaw  = raw.get("Yaw", 0.0) # horizontal rotation angle

        return {
            "voxelGrid":      voxelArr,
            "nearbyEntities": nearbyEntities,
            "life":           life,
            "pos":            np.array([xPos, zPos], dtype=np.float32),
            "yaw":            yaw,
        }

    def _emptyObs(self) -> dict:
        return {
            "voxelGrid":      np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32),
            "nearbyEntities": [],
            "life":           20.0,
            "pos":            np.zeros(2, dtype=np.float32),
            "yaw":            0.0,
        }

    def _getRewardsAll(self, obsAll: list[dict]) -> list[float]:
        # agent rewards computed manually to since there's no way to distinguish
        # player attack rewards in malmo (this way predator attacking predator isn't rewarded)
        preyIndices     = [2, 3]
        predatorIndices = [0, 1]

        healthDeltas = [
            self.prevHealth[i] - obsAll[i]["life"]
            for i in range(NUM_AGENTS)
        ]
        self.prevHealth = [obs["life"] for obs in obsAll]

        rewards = []
        for i in range(NUM_AGENTS):
            if i in predatorIndices:
                preyDamage = sum(max(0, healthDeltas[j]) for j in preyIndices)
                friendlyFire    = sum(max(0, healthDeltas[j]) for j in predatorIndices if j != i)
                rewards.append(preyDamage * 5 - friendlyFire * 5- 0.1)  # +5 per prey damage, -0.1 time penalty
            else:
                damageTaken = max(0, healthDeltas[i])
                rewards.append(0.1 - damageTaken * 5)  # +0.1 survival, -5 per damage taken

        return rewards

    def _getDonesAll(self) -> list[bool]:
        dones = []
        for host in self.agentHosts:
            worldState = host.getWorldState()
            dones.append(not worldState.is_mission_running)
        return dones

    @property
    def numActions(self) -> tuple[int, int, int]:
        """Returns (nMove, nTurn, nAttack) action counts."""
        return (len(MOVE_CMDS), len(TURN_CMDS), len(ATTACK_CMDS))

    @property
    def obsShape(self) -> dict:
        return {
            "voxelGrid": (GRID_SIZE * GRID_SIZE,),
            "pos":       (2,),
        }