import argparse
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from lander import VARIANTS, make_env
from lander import results
from lander.config import DEFAULT_STEPS, MODELS_DIR, N_ENVS, PPO_KWARGS, run_tag


class CurveLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.curve = []

    def _on_rollout_end(self):
        buffer = self.model.ep_info_buffer
        if buffer:
            self.curve.append([
                int(self.num_timesteps),
                round(float(np.mean([e["r"] for e in buffer])), 2),
                round(float(np.mean([e["l"] for e in buffer])), 1),
            ])

    def _on_step(self):
        return True


def train(variant, steps, seed):
    torch.set_num_threads(1)
    envs = make_vec_env(lambda: make_env(variant), n_envs=N_ENVS, seed=seed)
    model = PPO("MlpPolicy", envs, seed=seed, verbose=0, **PPO_KWARGS)
    logger = CurveLogger()

    start = time.time()
    model.learn(total_timesteps=steps, callback=logger)
    minutes = (time.time() - start) / 60

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / run_tag(variant, seed))
    parameters = sum(p.numel() for p in model.policy.parameters())

    results.save(
        variant, seed,
        steps=steps,
        parameters=parameters,
        train_minutes=round(minutes, 2),
        curve=logger.curve,
    )
    final = logger.curve[-1] if logger.curve else [steps, float("nan"), float("nan")]
    print(f"[{run_tag(variant, seed)}] {steps:,} steps in {minutes:.1f} min, "
          f"{parameters:,} parameters, final shaped ep_rew_mean {final[1]:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO on one reward variant.")
    parser.add_argument("variant", choices=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.variant, args.steps, args.seed)


if __name__ == "__main__":
    main()
