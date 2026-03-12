from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.qmixAgent import QMIX
from src.utils.obsUtils import batchObsAllAgents, buildGlobalState
from src.utils.replayBuffer import Episode, ReplayBuffer
from src.envs.malmoEnv import MalmoEnv, AGENT_NAMES
import torch, torch.nn.functional as F
import random, numpy as np

MAX_STEPS              = 1_000_000
EPISODE_LIMIT          = 500        # max timesteps per episode (Malmo has 60s = ~600 steps at 0.1s sleep)
BATCH_SIZE             = 32
HIDDEN_DIM             = 64
GAMMA                  = 0.99
EPSILON_START          = 1.0
EPSILON_END            = 0.05
EPSILON_DECAY          = 50_000     # steps over which epsilon anneals
TARGET_UPDATE_INTERVAL = 200        # episodes, not steps
LR                     = 1e-3
BUFFER_SIZE            = 1000       # max episodes in replay buffer
MIN_BUFFER_EPISODES    = 50         # don't train until buffer has this many episodes
CHECKPOINT_EPISODES = 25
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "qmix_latest.pt")
REPLAY_FILE = os.path.join(CHECKPOINT_DIR, "replay_latest.pkl")

N_AGENTS   = 4
OBS_DIM    = 69
STATE_DIM  = 16
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QMIX for Malmo predator-prey")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint if available")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_FILE, help="path to model checkpoint")
    parser.add_argument("--replay", type=str, default=REPLAY_FILE, help="path to replay buffer checkpoint")
    return parser.parse_args()


def saveTrainingState(qmix: QMIX, buffer: ReplayBuffer, checkpointPath: str, replayPath: str,
                      step: int, episode: int, epsilon: float, reason: str):
    extra = {
        "step": step,
        "episode": episode,
        "epsilon": epsilon,
    }
    qmix.saveCheckpoint(checkpointPath, extraState=extra)
    buffer.save(replayPath)
    print(f"[checkpoint] saved ({reason}) -> {checkpointPath}")


def maybeLoadTrainingState(qmix: QMIX, buffer: ReplayBuffer, checkpointPath: str,
                           replayPath: str, shouldResume: bool) -> tuple[int, int, float]:
    if not shouldResume:
        return 0, 0, EPSILON_START

    checkpoint = Path(checkpointPath)
    if not checkpoint.exists():
        print(f"[resume] checkpoint not found at {checkpointPath}, starting fresh")
        return 0, 0, EPSILON_START

    extra = qmix.loadCheckpoint(checkpointPath)
    bufferLoaded = buffer.load(replayPath)
    step = int(extra.get("step", 0))
    episode = int(extra.get("episode", 0))
    epsilon = float(extra.get("epsilon", EPSILON_START))
    print(
        f"[resume] loaded step={step}, episode={episode}, epsilon={epsilon:.3f}, "
        f"replayLoaded={bufferLoaded}"
    )
    return step, episode, epsilon


def main():
    args = parseArgs()
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.replay).parent.mkdir(parents=True, exist_ok=True)

    missionXml = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
    env = MalmoEnv(missionXml, episodeStepLimit=EPISODE_LIMIT)
    qmix = QMIX(N_AGENTS, OBS_DIM, STATE_DIM, HIDDEN_DIM, LR, DEVICE)
    buffer = ReplayBuffer(BUFFER_SIZE)

    step, episodeCount, epsilon = maybeLoadTrainingState(
        qmix, buffer, args.checkpoint, args.replay, args.resume
    )

    try:
        while step < MAX_STEPS:
            obs = env.reset()
            qmix.initHiddenStates()
            t = 0
            ep = Episode()

            while t < EPISODE_LIMIT:
                obsTensor = batchObsAllAgents(obs, AGENT_NAMES, torch.device(DEVICE))
                globalState = buildGlobalState(obs, AGENT_NAMES)

                # forward pass all agents
                with torch.no_grad():
                    moveQ, turnQ, attackQ, qmix.hiddenStates = qmix.agentNet(obsTensor, qmix.hiddenStates)

                # epsilon greedy action per agent
                actions = []
                for i in range(N_AGENTS):
                    if random.random() < epsilon:
                        action = (random.randint(0, 2), random.randint(0, 2), random.randint(0, 1))
                    else:
                        action = (
                            moveQ[i].argmax().item(),
                            turnQ[i].argmax().item(),
                            attackQ[i].argmax().item(),
                        )
                    actions.append(action)

                nextObs, rewards, dones = env.step(actions)

                ep.observations.append(obsTensor.cpu().numpy())
                ep.actions.append(np.array(actions))
                ep.rewards.append(np.array(rewards))
                ep.states.append(globalState)
                episodeDone = all(dones)
                ep.dones.append(episodeDone)

                obs = nextObs
                epsilon = max(
                    EPSILON_END,
                    EPSILON_START - (EPSILON_START - EPSILON_END) * (step / EPSILON_DECAY),
                )
                step += 1
                t += 1

                if episodeDone:
                    break

            buffer.addEpisode(ep)
            episodeCount += 1

            if len(buffer) >= MIN_BUFFER_EPISODES:
                episodes: list[Episode] = buffer.sample(BATCH_SIZE)
                totalLoss = torch.tensor(0.0, device=DEVICE)
                trainedEpisodes = 0
                qmix.optimizer.zero_grad()

                for episode in episodes:
                    T = len(episode)
                    if T < 2:
                        continue

                    trainHidden = torch.zeros(1, N_AGENTS, HIDDEN_DIM, device=DEVICE)
                    targetHidden = torch.zeros(1, N_AGENTS, HIDDEN_DIM, device=DEVICE)

                    episodeLoss = torch.tensor(0.0, device=DEVICE)
                    for t in range(T - 1):
                        obsTensor = torch.tensor(episode.observations[t], dtype=torch.float32, device=DEVICE)
                        state = torch.tensor(episode.states[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        actArr = episode.actions[t]
                        rewArr = episode.rewards[t]

                        # forward pass live network
                        moveQ, turnQ, attackQ, trainHidden = qmix.agentNet(obsTensor, trainHidden)

                        # pick Q value of action actually taken per agent
                        agentQvals = torch.stack([
                            moveQ[i][actArr[i, 0]] + turnQ[i][actArr[i, 1]] + attackQ[i][actArr[i, 2]]
                            for i in range(N_AGENTS)
                        ]).unsqueeze(0)

                        qTot = qmix.mixer(agentQvals, state)

                        # target network forward pass on next timestep
                        nextObsTensor = torch.tensor(
                            episode.observations[t + 1], dtype=torch.float32, device=DEVICE
                        )
                        nextState = torch.tensor(
                            episode.states[t + 1], dtype=torch.float32, device=DEVICE
                        ).unsqueeze(0)

                        with torch.no_grad():
                            nextMoveQ, nextTurnQ, nextAttackQ, targetHidden = qmix.targetAgentNet(
                                nextObsTensor, targetHidden
                            )
                            nextAgentQvals = torch.stack([
                                nextMoveQ[i].max() + nextTurnQ[i].max() + nextAttackQ[i].max()
                                for i in range(N_AGENTS)
                            ]).unsqueeze(0)

                            targetQtot = qmix.targetMixer(nextAgentQvals, nextState)

                            done = float(episode.dones[t])
                            reward = torch.tensor(rewArr.mean(), dtype=torch.float32, device=DEVICE)
                            yTot = reward + GAMMA * targetQtot * (1 - done)

                        episodeLoss += F.mse_loss(qTot, yTot.detach())

                    totalLoss += episodeLoss / T
                    trainedEpisodes += 1

                if trainedEpisodes > 0:
                    totalLoss = totalLoss / trainedEpisodes
                    totalLoss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(qmix.agentNet.parameters()) + list(qmix.mixer.parameters()), 10.0
                    )
                    qmix.optimizer.step()

            # target update
            if episodeCount % TARGET_UPDATE_INTERVAL == 0:
                qmix.updateTargets()

            # logging
            if step % 1000 == 0:
                meanReward = np.mean([np.sum(ep_i.rewards) for ep_i in buffer.buffer[-10:]])
                print(f"step={step} | epsilon={epsilon:.3f} | meanReward(last 10 ep)={meanReward:.2f}")

            if episodeCount % CHECKPOINT_EPISODES == 0:
                saveTrainingState(
                    qmix, buffer, args.checkpoint, args.replay,
                    step=step, episode=episodeCount, epsilon=epsilon,
                    reason=f"periodic ep={episodeCount}",
                )

        saveTrainingState(
            qmix, buffer, args.checkpoint, args.replay,
            step=step, episode=episodeCount, epsilon=epsilon,
            reason="training complete",
        )
    except KeyboardInterrupt:
        saveTrainingState(
            qmix, buffer, args.checkpoint, args.replay,
            step=step, episode=episodeCount, epsilon=epsilon,
            reason="keyboard interrupt",
        )
        print("Training interrupted with Ctrl+C. Checkpoint saved; exiting.")


if __name__ == "__main__":
    main()