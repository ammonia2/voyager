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
MOVE_CMDS = ["move 1", "move -1", "move 0"]
TURN_CMDS = ["turn -1", "turn 1", "turn 0"]
ATTACK_CMDS = ["attack 1", "attack 0"]

NUM_AGENTS = 4
AGENT_NAMES = ["Predator1", "Predator2", "Prey1", "Prey2"]
BASE_PORT = 10000
GRID_SIZE = 7  # could increase to 9x9 or decrease to 5x5
STEP_SLEEP = 0.1  # seconds between steps
RESET_WAIT_TIMEOUT = 30.0
CLIENT_POOL_COOLDOWN = 2.0
RESET_SETTLE_TIME = 0.3
DEFAULT_EPISODE_STEP_LIMIT = 500

PREDATOR_INDICES = [0, 1]
PREY_INDICES = [2, 3]
SPAWN_POINTS = [
    (5.0, 4.0, 5.0, 0.0),
    (15.0, 4.0, 5.0, 0.0),
    (5.0, 4.0, 15.0, 0.0),
    (15.0, 4.0, 15.0, 0.0),
]

BLOCK_TO_ID = {
    "air": 0,
    "stone": 1,
    "stonebrick": 2,
    "grass": 3,
    "dirt": 4,
    "cobblestone": 5,
    "sand": 6,
    "gravel": 7,
}
DEFAULT_BLOCK_ID = 15  # unknown block type


class MalmoEnv:
    def __init__(self, missionXmlPath: str, episodeStepLimit: int = DEFAULT_EPISODE_STEP_LIMIT):
        self.missionXml = Path(missionXmlPath).read_text()
        self.agentHosts = [MalmoPython.AgentHost() for _ in range(NUM_AGENTS)]
        self.clientPool = self._buildClientPool()
        self.prevHealth = []
        self.episodeStepLimit = episodeStepLimit
        self.episodeSteps = 0
        self.missionStarted = False

    def _buildClientPool(self) -> MalmoPython.ClientPool:
        pool = MalmoPython.ClientPool()
        for i in range(NUM_AGENTS):
            pool.add(MalmoPython.ClientInfo("127.0.0.1", BASE_PORT + i))
        return pool

    def reset(self) -> list[dict]:
        allRunning = self._allMissionsRunning()
        if not self.missionStarted or not allRunning:
            if self.missionStarted and not allRunning:
                print("[malmoEnv] Mission not running during reset; restarting full mission.")
            self._startMission()
        else:
            self._softResetEpisode()

        obsAll = self._getObsAll()
        self.prevHealth = [obs["life"] for obs in obsAll]
        self.episodeSteps = 0
        return obsAll

    def _startMission(self):
        self._ensureMissionStopped()
        time.sleep(CLIENT_POOL_COOLDOWN)

        mission = MalmoPython.MissionSpec(self.missionXml, True)
        missionRecord = MalmoPython.MissionRecordSpec()
        experimentId = str(int(time.time()))  # unique only when mission is created

        for i, host in enumerate(self.agentHosts):
            host.startMission(mission, self.clientPool, missionRecord, i, experimentId)
            if i == 0:
                time.sleep(30)  # role 0 needs time to start the server
            else:
                time.sleep(1)

        self._waitForAllAgents()
        self.missionStarted = True

    def _softResetEpisode(self):
        # Best-effort in-place reset: neutralize controls and restore spawn transforms.
        for i, host in enumerate(self.agentHosts):
            x, y, z, yaw = SPAWN_POINTS[i]
            for cmd in ("move 0", "turn 0", "attack 0", f"tp {x} {y} {z}", f"setYaw {yaw}"):
                try:
                    host.sendCommand(cmd)
                except RuntimeError:
                    pass
        time.sleep(RESET_SETTLE_TIME)

    def _allMissionsRunning(self) -> bool:
        return all(host.getWorldState().is_mission_running for host in self.agentHosts)

    def _ensureMissionStopped(self, timeout: float = RESET_WAIT_TIMEOUT):
        deadline = time.time() + timeout

        while time.time() < deadline:
            runningHosts = []
            for host in self.agentHosts:
                worldState = host.getWorldState()
                if worldState.is_mission_running:
                    runningHosts.append(host)

            if not runningHosts:
                return

            for host in runningHosts:
                try:
                    host.sendCommand("quit")
                except RuntimeError:
                    pass

            time.sleep(1.0)

        raise RuntimeError("Timed out waiting for the previous Malmo mission to stop.")

    def step(self, actions: list[tuple[int, int, int]]) -> tuple[list, list, list]:
        """
        actions: list of (moveIdx, turnIdx, attackIdx) per agent
        returns: (obsAll, rewardsAll, donesAll)
        """
        for host, action in zip(self.agentHosts, actions):
            moveIdx, turnIdx, attackIdx = action
            host.sendCommand(MOVE_CMDS[moveIdx])
            host.sendCommand(TURN_CMDS[turnIdx])
            host.sendCommand(ATTACK_CMDS[attackIdx])

        time.sleep(STEP_SLEEP)

        obsAll = self._getObsAll()
        
        # Check for accidentally dead predators and respawn them instantly
        for i in PREDATOR_INDICES:
            if obsAll[i]["life"] <= 0:
                x, y, z, yaw = SPAWN_POINTS[i]
                
                try:
                    # tp and setHealth restore the agent immediately
                    self.agentHosts[i].sendCommand(f"tp {x} {y} {z}")
                    self.agentHosts[i].sendCommand(f"setYaw {yaw}")
                    self.agentHosts[i].sendCommand("setHealth 20")
                except RuntimeError:
                    pass
                
                # Optimistically update the obs
                obsAll[i]["life"] = 20.0
                obsAll[i]["pos"] = np.array([x, z], dtype=np.float32)
                obsAll[i]["yaw"] = yaw

        rewardsAll = self._getRewardsAll(obsAll)
        self.episodeSteps += 1
        donesAll = self._getDonesAll(obsAll)

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
            dtype=np.float32,
        )

        # Nearby entities: list of {name, x, y, z, life}
        nearbyEntities = raw.get("nearbyEntities", [])

        # Agent stats
        life = raw.get("Life", 20.0)
        xPos = raw.get("XPos", 0.0)
        zPos = raw.get("ZPos", 0.0)
        yaw = raw.get("Yaw", 0.0)  # horizontal rotation angle

        return {
            "voxelGrid": voxelArr,
            "nearbyEntities": nearbyEntities,
            "life": life,
            "pos": np.array([xPos, zPos], dtype=np.float32),
            "yaw": yaw,
        }

    def _emptyObs(self) -> dict:
        return {
            "voxelGrid": np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32),
            "nearbyEntities": [],
            "life": 20.0,
            "pos": np.zeros(2, dtype=np.float32),
            "yaw": 0.0,
        }

    def _getRewardsAll(self, obsAll: list[dict]) -> list[float]:
        # Agent rewards computed manually because Malmo cannot attribute
        # damage reward to the attacker when all entities are players.
        healthDeltas = [self.prevHealth[i] - obsAll[i]["life"] for i in range(NUM_AGENTS)]
        self.prevHealth = [obs["life"] for obs in obsAll]

        rewards = []
        for i in range(NUM_AGENTS):
            if i in PREDATOR_INDICES:
                preyDamage = sum(max(0, healthDeltas[j]) for j in PREY_INDICES)
                friendlyFire = sum(max(0, healthDeltas[j]) for j in PREDATOR_INDICES if j != i)
                rewards.append(preyDamage * 5 - friendlyFire * 5 - 0.1)
            else:
                damageTaken = max(0, healthDeltas[i])
                rewards.append(0.1 - damageTaken * 5)

        return rewards

    def _getDonesAll(self, obsAll: list[dict]) -> list[bool]:
        missionDropped = not self._allMissionsRunning()
        if missionDropped:
            print(f"[malmoEnv] Mission dropped at episodeStep={self.episodeSteps}; forcing episode end.")
            self.missionStarted = False

        preyAlive = any(obsAll[i]["life"] > 0 for i in PREY_INDICES)
        predatorAlive = any(obsAll[i]["life"] > 0 for i in PREDATOR_INDICES)
        teamEliminated = (not preyAlive) or (not predatorAlive)
        timeUp = self.episodeSteps >= self.episodeStepLimit
        done = missionDropped or teamEliminated or timeUp

        return [done] * NUM_AGENTS

    @property
    def numActions(self) -> tuple[int, int, int]:
        """Returns (nMove, nTurn, nAttack) action counts."""
        return (len(MOVE_CMDS), len(TURN_CMDS), len(ATTACK_CMDS))

    @property
    def obsShape(self) -> dict:
        return {
            "voxelGrid": (GRID_SIZE * GRID_SIZE,),
            "pos": (2,),
        }