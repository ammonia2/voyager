from __future__ import annotations
import MalmoPython
import json
import time
import numpy as np
from pathlib import Path

MOVE_CMDS   = ["move 1", "move -1", "move 0"]
TURN_CMDS   = ["turn -1", "turn 1", "turn 0"]
ATTACK_CMDS = ["attack 1", "attack 0"]

NUM_AGENTS       = 3
AGENT_NAMES      = ["Predator1", "Predator2", "Prey1"]
PREDATOR_INDICES = [0, 1]
PREY_INDICES     = [2]
SPAWN_POINTS     = [
    (5.0, 4.0, 5.0, 0.0),
    (15.0, 4.0, 5.0, 0.0),
    (10.0, 4.0, 15.0, 0.0),
]
BASE_PORT        = 10000
GRID_SIZE        = 7
STEP_SLEEP       = 0.1
RESET_SETTLE_TIME = 0.3
RESET_WAIT_TIMEOUT = 30.0
CLIENT_POOL_COOLDOWN = 2.0

ARENA_MIN = 1.0
ARENA_MAX = 19.0

BLOCK_TO_ID = {
    'air': 0, 'stone': 1, 'stonebrick': 2, 'grass': 3,
    'dirt': 4, 'cobblestone': 5, 'sand': 6, 'gravel': 7,
}
DEFAULT_BLOCK_ID = 15


class MalmoEnv:
    def __init__(self, missionXmlPath: str):
        self.missionXml  = Path(missionXmlPath).read_text()
        self.agentHosts  = [MalmoPython.AgentHost() for _ in range(NUM_AGENTS)]
        self.clientPool  = self._buildClientPool()
        self.prevHealth  = []
        self.missionStarted = False

    def _buildClientPool(self) -> MalmoPython.ClientPool:
        pool = MalmoPython.ClientPool()
        for i in range(NUM_AGENTS):
            pool.add(MalmoPython.ClientInfo("127.0.0.1", BASE_PORT + i))
        return pool

    def reset(self) -> list[dict]:
        if not self.missionStarted or not self._allMissionsRunning():
            self._startMission()
        else:
            resetOk = self._softResetEpisode()
            if not resetOk:
                self._restartMission()

        self._waitForAllAgents()
        obsAll          = self._getObsAll()
        self.prevHealth = [obs["life"] for obs in obsAll]
        return obsAll

    def _restartMission(self):
        self._ensureMissionStopped()
        time.sleep(CLIENT_POOL_COOLDOWN)
        self._startMission()

    def _startMission(self):
        mission       = MalmoPython.MissionSpec(self.missionXml, True)
        missionRecord = MalmoPython.MissionRecordSpec()
        experimentId  = str(int(time.time()))

        for i, host in enumerate(self.agentHosts):
            host.startMission(mission, self.clientPool, missionRecord, i, experimentId)
            if i == 0:
                time.sleep(30)
            else:
                time.sleep(1)

        self.missionStarted = True

    def _softResetEpisode(self) -> bool:
        for _ in range(3):
            for i, host in enumerate(self.agentHosts):
                x, y, z, yaw = SPAWN_POINTS[i]
                for cmd in ("move 0", "turn 0", "attack 0", f"tp {x} {y} {z}", f"setYaw {yaw}"):
                    try:
                        host.sendCommand(cmd)
                    except RuntimeError:
                        pass

            time.sleep(RESET_SETTLE_TIME)

            obsAll = self._getObsAll()
            if self._isResetStateHealthy(obsAll):
                return True

        return False

    def _allMissionsRunning(self) -> bool:
        return all(host.getWorldState().is_mission_running for host in self.agentHosts)

    def _ensureMissionStopped(self, timeout: float = RESET_WAIT_TIMEOUT):
        deadline = time.time() + timeout

        while time.time() < deadline:
            runningHosts = []
            for host in self.agentHosts:
                ws = host.getWorldState()
                if ws.is_mission_running:
                    runningHosts.append(host)

            if not runningHosts:
                self.missionStarted = False
                return

            for host in runningHosts:
                try:
                    host.sendCommand("quit")
                except RuntimeError:
                    pass

            time.sleep(1.0)

        raise RuntimeError("Timed out waiting for previous mission to stop.")

    def _isResetStateHealthy(self, obsAll: list[dict]) -> bool:
        if len(obsAll) != NUM_AGENTS:
            return False

        for obs in obsAll:
            if obs["life"] <= 0:
                return False

            x, z = float(obs["pos"][0]), float(obs["pos"][1])
            if not (np.isfinite(x) and np.isfinite(z)):
                return False
            if x < ARENA_MIN or x > ARENA_MAX or z < ARENA_MIN or z > ARENA_MAX:
                return False

        return self._allMissionsRunning()

    def step(self, actions: list[tuple[int, int, int]]) -> tuple[list, list, list]:
        for host, action in zip(self.agentHosts, actions):
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
            ws = host.getWorldState()
            while not ws.has_mission_begun:
                time.sleep(0.1)
                ws = host.getWorldState()

    def _getObsAll(self) -> list[dict]:
        obs = []
        for host in self.agentHosts:
            ws = host.getWorldState()
            if ws.number_of_observations_since_last_state > 0:
                raw = json.loads(ws.observations[-1].text)
                obs.append(self._parseObs(raw))
            else:
                obs.append(self._emptyObs())
        return obs

    def _parseObs(self, raw: dict) -> dict:
        voxelArr = np.array(
            [BLOCK_TO_ID.get(b, DEFAULT_BLOCK_ID) for b in raw.get("voxelObs", [])],
            dtype=np.float32,
        )
        return {
            "voxelGrid":      voxelArr,
            "nearbyEntities": raw.get("nearbyEntities", []),
            "life":           raw.get("Life", 20.0),
            "pos":            np.array([raw.get("XPos", 0.0), raw.get("ZPos", 0.0)], dtype=np.float32),
            "yaw":            raw.get("Yaw", 0.0),
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
        healthDeltas    = [self.prevHealth[i] - obsAll[i]["life"] for i in range(NUM_AGENTS)]
        self.prevHealth = [obs["life"] for obs in obsAll]

        rewards = []
        for i in range(NUM_AGENTS):
            if i in PREDATOR_INDICES:
                preyDamage   = max(0, healthDeltas[PREY_INDICES[0]])
                friendlyFire = sum(max(0, healthDeltas[j]) for j in PREDATOR_INDICES if j != i)
                rewards.append(preyDamage * 5 - friendlyFire * 5 - 0.1)
            else:
                rewards.append(0.1 - max(0, healthDeltas[i]) * 5)

        return rewards

    def _getDonesAll(self) -> list[bool]:
        dones = [not host.getWorldState().is_mission_running for host in self.agentHosts]
        if any(dones):
            self.missionStarted = False
        return dones

    @property
    def numActions(self) -> tuple[int, int, int]:
        return (len(MOVE_CMDS), len(TURN_CMDS), len(ATTACK_CMDS))