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
SELF_AGENT_IDX   = 2
BASE_PORT        = 10000
GRID_SIZE        = 7
STEP_SLEEP       = 0.1
RESET_SETTLE_TIME = 1.0
RESET_WAIT_TIMEOUT = 60.0
CLIENT_POOL_COOLDOWN = 5.0
DEFAULT_EPISODE_STEP_LIMIT = 50

ARENA_MIN = 2.0
ARENA_MAX = 18.0
SPAWN_Y   = 4.0
MIN_SPAWN_DIST = 8.0


def decode_flat_action(flat_idx: int) -> tuple[int, int, int]:
    """
    Convert flat action index [0-17] to (move, turn, attack).
    Encoding: flat = move*6 + turn*2 + attack
    """
    attack = flat_idx % 2
    turn   = (flat_idx // 2) % 3
    move   = flat_idx // 6
    return move, turn, attack


def _randomSpawnPoints() -> list[tuple[float, float, float, float]]:
    positions: list[tuple[float, float]] = []
    max_tries = 200
    for _ in range(NUM_AGENTS):
        for _ in range(max_tries):
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


BLOCK_TO_ID = {
    'air': 0, 'stone': 1, 'stonebrick': 2, 'grass': 3,
    'dirt': 4, 'cobblestone': 5, 'sand': 6, 'gravel': 7,
}
DEFAULT_BLOCK_ID = 15


class MalmoEnv:
    def __init__(self, missionXmlPath: str, episodeStepLimit: int = DEFAULT_EPISODE_STEP_LIMIT):
        self.missionXml     = Path(missionXmlPath).read_text()
        self.agentHosts     = [MalmoPython.AgentHost() for _ in range(NUM_AGENTS)]
        self.clientPool     = self._buildClientPool()
        self.prevHealth     = []
        self.missionStarted = False
        self.preyWasTagged  = False
        self.episodeSteps   = 0
        self.episodeStepLimit = episodeStepLimit
        self._spawnPoints: list[tuple[float, float, float, float]] = _randomSpawnPoints()

    def _buildClientPool(self) -> MalmoPython.ClientPool:
        pool = MalmoPython.ClientPool()
        for i in range(NUM_AGENTS):
            pool.add(MalmoPython.ClientInfo("127.0.0.1", BASE_PORT + i))
        return pool

    def reset(self) -> list[dict]:
        for reset_attempt in range(3):
            try:
                self._spawnPoints  = _randomSpawnPoints()
                self.preyWasTagged = False
                self.episodeSteps  = 0

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
            except RuntimeError as e:
                print(f"[malmoEnv] Reset attempt {reset_attempt+1} failed: {e}", flush=True)
                if reset_attempt < 2:
                    print(f"[malmoEnv] Waiting 5s before retry...", flush=True)
                    time.sleep(5.0)
                    self.missionStarted = False  # Force full restart on next attempt
                else:
                    raise RuntimeError(f"Failed to reset environment after 3 attempts: {e}")

    def _restartMission(self):
        for attempt in range(3):
            try:
                self._ensureMissionStopped()
                time.sleep(CLIENT_POOL_COOLDOWN)
                self._startMission()
                return
            except RuntimeError:
                print(f"[malmoEnv] Restart attempt {attempt+1} failed, retrying...", flush=True)
                time.sleep(5.0)
        raise RuntimeError("Failed to restart mission after 3 attempts.")

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

    RESET_GRACE_STEPS = 2

    def _softResetEpisode(self) -> bool:
        for _ in range(3):
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
                h for h in self.agentHosts
                if h.getWorldState().is_mission_running
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
            x, z = float(obs["x"]), float(obs["z"])
            if not (np.isfinite(x) and np.isfinite(z)):
                return False
            if x < ARENA_MIN or x > ARENA_MAX or z < ARENA_MIN or z > ARENA_MAX:
                return False
        return self._allMissionsRunning()

    def step(self, actions: list[int]) -> tuple[list[dict], list[float], list[bool], dict]:
        """
        actions: list of 3 flat action indices [act0, act1, act2], each in [0..17]
        returns: (obs_all, rewards, dones, info)
        """
        assert len(actions) == NUM_AGENTS, \
            f"Expected {NUM_AGENTS} actions, got {len(actions)}"

        for host, flat_idx in zip(self.agentHosts, actions):
            move, turn, attack = decode_flat_action(int(flat_idx))
            host.sendCommand(MOVE_CMDS[move])
            host.sendCommand(TURN_CMDS[turn])
            host.sendCommand(ATTACK_CMDS[attack])

        time.sleep(STEP_SLEEP)

        try:
            obsAll     = self._getObsAll()
            rewardsAll = self._getRewardsAll(obsAll)
            donesAll   = self._getDonesAll()
            self.episodeSteps += 1
            return obsAll, rewardsAll, donesAll, {}
        except Exception as e:
            print(f"[malmoEnv] Step failed: {e}. Ending episode.", flush=True)
            # Return current obs and mark as done to avoid total crash
            self.preyWasTagged = True
            donesAll = [True] * NUM_AGENTS
            self.episodeSteps += 1
            return self._getObsAll() if len(self.prevHealth) > 0 else [self._emptyObs() for _ in range(NUM_AGENTS)], \
                   [0.0] * NUM_AGENTS, donesAll, {"error": str(e)}

    def _waitForAllAgents(self):
        max_wait = 60.0
        for i, host in enumerate(self.agentHosts):
            start_time = time.time()
            ws = host.getWorldState()
            while not ws.has_mission_begun:
                elapsed = time.time() - start_time
                if elapsed > max_wait:
                    raise RuntimeError(f"Agent {i} mission did not start within {max_wait}s")
                time.sleep(0.1)
                ws = host.getWorldState()

    def _getObsAll(self) -> list[dict]:
        obs = []
        for i, host in enumerate(self.agentHosts):
            ws = host.getWorldState()
            if ws.number_of_observations_since_last_state > 0:
                raw = json.loads(ws.observations[-1].text)
                obs.append(self._parseObs(raw, i))
            else:
                obs.append(self._emptyObs())
        return obs

    def _parseObs(self, raw: dict, agent_idx: int) -> dict:
        """
        Parse raw Malmo JSON into OMIS-compatible observation dict.

        Keys returned:
            "ob"       : list of 49 ints  — flattened 7x7 voxel block IDs
            "entities" : list of [x, z, yaw, health, is_same_team]
            "x"        : float
            "z"        : float
            "yaw"      : float
            "life"     : float
        """
        # Voxel grid — 49 block IDs
        voxel_raw = raw.get("voxelObs", [])
        ob = [BLOCK_TO_ID.get(b, DEFAULT_BLOCK_ID) for b in voxel_raw]
        while len(ob) < GRID_SIZE * GRID_SIZE:
            ob.append(0)
        ob = ob[:GRID_SIZE * GRID_SIZE]

        # Agent stats
        x    = float(raw.get("XPos", 0.0))
        z    = float(raw.get("ZPos", 0.0))
        yaw  = float(raw.get("Yaw",  0.0))
        life = float(raw.get("Life", 20.0))

        # Nearby entities → [x, z, yaw, health, is_same_team]
        entities = []
        for e in raw.get("nearbyEntities", []):
            e_name = e.get("name", "")
            e_x    = float(e.get("x",    0.0))
            e_z    = float(e.get("z",    0.0))
            e_yaw  = float(e.get("yaw",  0.0))
            e_life = float(e.get("life", 20.0))

            if agent_idx in PREDATOR_INDICES:
                is_same_team = 1.0 if e_name in ["Predator1", "Predator2"] else 0.0
            else:
                is_same_team = 0.0

            entities.append([e_x, e_z, e_yaw, e_life, is_same_team])

        return {
            "ob":       ob,
            "entities": entities,
            "x":        x,
            "z":        z,
            "yaw":      yaw,
            "life":     life,
        }

    def _emptyObs(self) -> dict:
        return {
            "ob":       [0] * (GRID_SIZE * GRID_SIZE),
            "entities": [],
            "x":        0.0,
            "z":        0.0,
            "yaw":      0.0,
            "life":     20.0,
        }

    MELEE_RANGE  = 3.5
    TAG_REWARD   = 10.0
    TAG_PENALTY  = -10.0
    TIME_PENALTY = 0.3

    def _getRewardsAll(self, obsAll: list[dict]) -> list[float]:
        healthDeltas    = [self.prevHealth[i] - obsAll[i]["life"] for i in range(NUM_AGENTS)]
        self.prevHealth = [obs["life"] for obs in obsAll]

        preyPos  = np.array([obsAll[PREY_INDICES[0]]["x"], obsAll[PREY_INDICES[0]]["z"]])
        preyHit  = (healthDeltas[PREY_INDICES[0]] > 0
                    and self.episodeSteps > self.RESET_GRACE_STEPS)
        self.preyWasTagged = preyHit

        rewards = []
        for i in range(NUM_AGENTS):
            if i in PREDATOR_INDICES:
                if preyHit:
                    predPos  = np.array([obsAll[i]["x"], obsAll[i]["z"]])
                    dist     = float(np.linalg.norm(predPos - preyPos))
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

        timeUp = self.episodeSteps >= self.episodeStepLimit
        done   = missionDead or self.preyWasTagged or timeUp
        return [done] * NUM_AGENTS

    @property
    def numActions(self) -> int:
        return len(MOVE_CMDS) * len(TURN_CMDS) * len(ATTACK_CMDS)  # 18
