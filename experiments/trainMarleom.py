from __future__ import annotations
import time
import random
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.malmoEnv2 import MalmoEnv, MOVE_CMDS, TURN_CMDS, ATTACK_CMDS
from src.agents.marleom import MARLeOM
from src.utils.logs import MARLLogger
from src.utils.replayBuffer import ReplayBuffer
from src.utils.obsUtils import flattenObsAll, NUM_AGENTS, PREDATOR_INDICES, PREY_INDICES

# ---- Hyperparameters ----
MISSION_XML     = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
NUM_EPISODES    = 500
MAX_STEPS       = 200
BUFFER_CAPACITY = 100_000
BATCH_SIZE      = 256
WARMUP_STEPS    = 1000
UPDATE_EVERY    = 4
LR              = 1e-3
ALPHA           = 0.2
GAMMA           = 0.95
TAU             = 0.01
DEVICE          = "cpu"
SAVE_EVERY      = 50
CKPT_DIR        = "checkpoints"
LOG_INTERVAL    = 10

# Debug flag: set to 0.0 to disable L1 opponent model (use pure L0 behavior cloning)
LEVEL_MIX       = 0.5

NEUTRAL_ACTION  = (2, 2, 1)


def randomAction() -> tuple[int, int, int]:
    return (
        random.randint(0, len(MOVE_CMDS) - 1),
        random.randint(0, len(TURN_CMDS) - 1),
        random.randint(0, len(ATTACK_CMDS) - 1),
    )


def preyPolicy() -> tuple[int, int, int]:
    """Scripted prey: move forward, random turn, never attack."""
    return (0, random.randint(0, 2), 1)


def diagnoseActionCollapse(flatObs: np.ndarray, agent: MARLeOM) -> str:
    """
    Check if predator actions have collapsed to degenerate pattern.
    Returns diagnostic string.
    """
    debug_info = agent.getActionDist(flatObs)
    
    diagnostics = []
    
    # Check opponent model prediction
    opp_pred = debug_info["opponent_pred"]
    pred_move = opp_pred[:1]  # First action is move
    prey_predicted_static = (opp_pred[0] > 0.7)  # One-hot or near one-hot on same action
    
    # Check actor move probabilities
    for idx, actor_dist in enumerate(debug_info["actors"]):
        moveP = actor_dist["moveP"]
        turnP = actor_dist["turnP"]
        attackP = actor_dist["attackP"]
        
        # Calculate entropy as proxy for diversity
        move_entropy = -np.sum(moveP[moveP > 0] * np.log(moveP[moveP > 0] + 1e-8))
        attack_entropy = -np.sum(attackP[attackP > 0] * np.log(attackP[attackP > 0] + 1e-8))
        
        is_collapsed = (move_entropy < 0.1) and (attackP[-1] > 0.8)
        diagnostics.append(f"  Agent {idx}: move_H={move_entropy:.3f}, attack_prob[punch]={attackP[-1]:.3f}")
        
        if is_collapsed:
            diagnostics.append(f"    ^^ DEGENERATE: collapsed to spin+punch")
    
    if prey_predicted_static:
        diagnostics.append(f"  Opponent model: predicts prey static (possible root cause)")
    
    return "\n".join(diagnostics)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    logger = MARLLogger(algo_name="MARLeOM", log_interval=LOG_INTERVAL, seed=0)

    env    = MalmoEnv(MISSION_XML)
    agent  = MARLeOM(lr=LR, alpha=ALPHA, gamma=GAMMA, tau=TAU, device=DEVICE, levelMix=LEVEL_MIX)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    totalSteps = 0

    for episode in range(1, NUM_EPISODES + 1):
        t0             = time.time()
        obsAll         = env.reset()
        lastActionsAll = [NEUTRAL_ACTION] * NUM_AGENTS
        flatObs        = flattenObsAll(obsAll, lastActionsAll)

        episodeRewardPred = 0.0
        episodeRewardPrey = 0.0
        stepCount         = 0

        for _ in range(MAX_STEPS):
            if totalSteps < WARMUP_STEPS:
                actions = [randomAction() for _ in range(NUM_AGENTS)]
            else:
                actions = agent.selectActions(flatObs, explore=True)
                actions[PREY_INDICES[0]] = preyPolicy()

            nextObsAll, rewardsAll, donesAll = env.step(actions)
            nextFlatObs = flattenObsAll(nextObsAll, actions)

            buffer.push(
                obsAll     = flatObs,
                actionsAll = np.array(actions,    dtype=np.int64),
                rewardsAll = np.array(rewardsAll, dtype=np.float32),
                nextObsAll = nextFlatObs,
                dones      = np.array(donesAll,   dtype=np.float32),
            )

            flatObs        = nextFlatObs
            obsAll         = nextObsAll
            totalSteps    += 1
            stepCount     += 1

            episodeRewardPred += sum(rewardsAll[i] for i in PREDATOR_INDICES)
            episodeRewardPrey += rewardsAll[PREY_INDICES[0]]

            if totalSteps >= WARMUP_STEPS and totalSteps % UPDATE_EVERY == 0:
                if len(buffer) >= BATCH_SIZE:
                    losses = agent.update(buffer.sample(BATCH_SIZE))
                    log_kwargs = {
                        "step": totalSteps,
                        "critic_loss": losses["criticLoss"],
                        "actor_loss": losses["actorLoss"],
                        "opponent_model_loss": losses["omLoss"],
                    }
                    if "om0Loss" in losses:
                        log_kwargs["oppModelL0_loss"] = losses["om0Loss"]
                    if "om1Loss" in losses:
                        log_kwargs["oppModelL1_loss"] = losses["om1Loss"]
                    logger.log_step(**log_kwargs)

            if all(donesAll):
                break

        teamWon = int(episodeRewardPred > episodeRewardPrey)
        
        # Diagnostic logging every 5 episodes after warmup
        if episode >= WARMUP_STEPS // MAX_STEPS and episode % 5 == 0:
            diag = diagnoseActionCollapse(flatObs, agent)
            print(f"\n[Ep {episode:4d}] Diagnostic:\n{diag}")
        
        logger.log_episode(
            episode=episode,
            episode_return=episodeRewardPred,
            win=teamWon,
            opponent_type="seen",
            prey_return=episodeRewardPrey,
            episode_steps=stepCount,
            buffer_size=len(buffer),
            total_steps=totalSteps,
            episode_seconds=time.time() - t0,
        )

        if episode % SAVE_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"marleom_ep{episode}.pt")
            agent.save(path)
            logger.log_step(totalSteps, checkpoint_saved=float(episode))

    agent.save(os.path.join(CKPT_DIR, "marleom_final.pt"))
    logger.print_final_summary()
    print("Training complete.")


if __name__ == "__main__":
    main()