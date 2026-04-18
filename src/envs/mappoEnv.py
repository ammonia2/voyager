from __future__ import annotations
import MalmoPython
import json
import time
import random
import numpy as np
from pathlib import Path

MOVE_CMDS   = ["move 1", "move -1", "move 0"]
TURN_CMDS   = ["turn -1", "turn 1", "turn 0"]
ATTACK_CMDS = ["attack 1", "attack 0"]

NUM_AGENTS       = 3
AGENT_NAMES      = ["Predator1", "Predator2", "Prey1"]
PREDATOR_INDICES = [0, 1]
PREY_INDICES     = [2]
BASE_PORT        = 10000

STEP_SLEEP         = 0.1
RESET_SETTLE_TIME  = 0.3
RESET_WAIT_TIMEOUT = 120.0
CLIENT_POOL_COOLDOWN  = 2.0
SOFT_RESET_MAX_TRIES  = 12
SOFT_RESET_FALLBACK_WINDOW = 8.0

ARENA_MIN      = 2.0
ARENA_MAX      = 18.0
SPAWN_Y        = 4.0
MIN_SPAWN_DIST = 8.0

BLOCK_TO_ID = {
    'air': 0, 'stone': 1, 'stonebrick': 2, 'grass': 3,
    'dirt': 4, 'cobblestone': 5, 'sand': 6, 'gravel': 7,
}
DEFAULT_BLOCK_ID = 15
VOXEL_GRID_DIM   = 5  # 5x5x5


def _randomSpawnPoints() -> list[tuple[float, float, float, float]]:
    positions: list[tuple[float, float]] = []
    for _ in range(NUM_AGENTS):
        for _ in range(200):
            x = random.uniform(ARENA_MIN, ARENA_MAX)
            z = random.uniform(ARENA_MIN, ARENA_MAX)
            if all(
                ((x - px) ** 2 + (z - pz) ** 2) ** 0.5 >= MIN_SPAWN_DIST
                for px, pz in positions
            ):
                positions.append((x, z))
                break
        else:
            positions.append((
                random.uniform(ARENA_MIN, ARENA_MAX),
                random.uniform(ARENA_MIN, ARENA_MAX),
            ))
    return [
        (x, SPAWN_Y, z, random.uniform(0.0, 360.0))
        for x, z in positions
    ]


class MalmoEnv:
    def __init__(self, missionXmlPath: str, portOffset: int = 0):
        self.missionXml  = Path(missionXmlPath).read_text()
        self.portOffset  = portOffset
        self.agentHosts  = [MalmoPython.AgentHost() for _ in range(NUM_AGENTS)]
        self.clientPool  = self._buildClientPool()
        self.prevHealth     = []
        self.missionStarted = False
        self.preyWasTagged  = False
        self.episodeSteps   = 0
        self._spawnPoints: list[tuple[float, float, float, float]] = _randomSpawnPoints()

    def _buildClientPool(self) -> MalmoPython.ClientPool:
        pool = MalmoPython.ClientPool()
        for i in range(NUM_AGENTS):
            pool.add(MalmoPython.ClientInfo("127.0.0.1", BASE_PORT + self.portOffset + i))
        return pool

    def reset(self) -> list[dict]:
        self._spawnPoints  = _randomSpawnPoints()
        self.preyWasTagged = False
        self.episodeSteps  = 0

        running = self.missionStarted and self._allMissionsRunning()
        if not running:
            self._restartMission()
        else:
            resetOk = self._softResetEpisode(maxTries=SOFT_RESET_MAX_TRIES)
            if not resetOk:
                deadline = time.time() + SOFT_RESET_FALLBACK_WINDOW
                while time.time() < deadline and self._allMissionsRunning():
                    if self._softResetEpisode(maxTries=2):
                        resetOk = True
                        break
                if not resetOk and not self._allMissionsRunning():
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
        maxRetries    = 3

        for attempt in range(maxRetries):
            experimentId = str(int(time.time() * 1000))
            try:
                for i, host in enumerate(self.agentHosts):
                    host.startMission(mission, self.clientPool, missionRecord, i, experimentId)
                    if i == 0:
                        time.sleep(30)
                    else:
                        time.sleep(1)
                self.missionStarted = True
                return
            except RuntimeError:
                if attempt == maxRetries - 1:
                    raise
                time.sleep(CLIENT_POOL_COOLDOWN)

    RESET_GRACE_STEPS = 2

    def _softResetEpisode(self, maxTries: int = 3) -> bool:
        for _ in range(maxTries):
            if not self._allMissionsRunning():
                return False
            for i, host in enumerate(self.agentHosts):
                x, y, z, yaw = self._spawnPoints[i]
                for cmd in ("attack 0", "move 0", "turn 0",
                            f"tp {x} {y} {z}", f"setYaw {yaw}", "setHealth 20",
                            "attack 0"):
                    try:
                        host.sendCommand(cmd)
                    except RuntimeError:
                        pass
            time.sleep(RESET_SETTLE_TIME * 2)
            obsAll = self._getObsAll()
            if self._isResetStateHealthy(obsAll):
                return True
        return False

    def _allMissionsRunning(self) -> bool:
        return all(host.getWorldState().is_mission_running for host in self.agentHosts)

    def _ensureMissionStopped(self, timeout: float = RESET_WAIT_TIMEOUT):
        deadline = time.time() + timeout
        while time.time() < deadline:
            runningHosts = [
                host for host in self.agentHosts
                if host.getWorldState().is_mission_running
            ]
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
        rawGrid  = raw.get("voxelObs", [])
        voxelArr = np.array(
            [BLOCK_TO_ID.get(b, DEFAULT_BLOCK_ID) for b in rawGrid],
            dtype=np.int32,
        )
        # Pad/truncate to exactly 5*5*5 = 125 elements
        expected = VOXEL_GRID_DIM ** 3
        if len(voxelArr) < expected:
            voxelArr = np.concatenate([voxelArr, np.zeros(expected - len(voxelArr), dtype=np.int32)])
        else:
            voxelArr = voxelArr[:expected]

        return {
            "voxelGrid":      voxelArr,
            "nearbyEntities": raw.get("nearbyEntities", []),
            "life":           raw.get("Life", 20.0),
            "pos":            np.array([raw.get("XPos", 0.0), raw.get("ZPos", 0.0)], dtype=np.float32),
            "yaw":            raw.get("Yaw", 0.0),
        }

    def _emptyObs(self) -> dict:
        return {
            "voxelGrid":      np.zeros(VOXEL_GRID_DIM ** 3, dtype=np.int32),
            "nearbyEntities": [],
            "life":           20.0,
            "pos":            np.zeros(2, dtype=np.float32),
            "yaw":            0.0,
        }

    MELEE_RANGE  = 3.5
    TAG_REWARD   = 10.0
    TAG_PENALTY  = -10.0
    TIME_PENALTY = 0.3

    def _getRewardsAll(self, obsAll: list[dict]) -> list[float]:
        healthDeltas    = [self.prevHealth[i] - obsAll[i]["life"] for i in range(NUM_AGENTS)]
        self.prevHealth = [obs["life"] for obs in obsAll]
        self.episodeSteps += 1

        preyPos = obsAll[PREY_INDICES[0]]["pos"]
        preyHit = (healthDeltas[PREY_INDICES[0]] > 0
                   and self.episodeSteps > self.RESET_GRACE_STEPS)
        self.preyWasTagged = preyHit

        rewards = []
        for i in range(NUM_AGENTS):
            if i in PREDATOR_INDICES:
                if preyHit:
                    dist      = float(np.linalg.norm(obsAll[i]["pos"] - preyPos))
                    tagCredit = self.TAG_REWARD if dist <= self.MELEE_RANGE else 0.0
                else:
                    tagCredit = 0.0
                rewards.append(tagCredit - self.TIME_PENALTY)
            else:
                rewards.append(self.TAG_PENALTY if preyHit else 0.1)
        return rewards

    def _getDonesAll(self) -> list[bool]:
        missionDead = not self._allMissionsRunning()
        if missionDead:
            self.missionStarted = False
        done = missionDead or self.preyWasTagged
        return [done] * NUM_AGENTS

    @property
    def numActions(self) -> tuple[int, int, int]:
        return (len(MOVE_CMDS), len(TURN_CMDS), len(ATTACK_CMDS))