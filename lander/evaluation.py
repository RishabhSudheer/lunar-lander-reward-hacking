import numpy as np

from .config import EVAL_EPISODES, EVAL_SEED_START
from .rewards import make_env


def classify(terminated, last_reward):
    if terminated and last_reward == 100:
        return "landed"
    if terminated and last_reward == -100:
        return "crashed"
    return "timed_out"


def evaluate(model, episodes=EVAL_EPISODES, seed_start=EVAL_SEED_START):
    env = make_env("baseline")
    scores, lengths, outcomes = [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed_start + episode)
        total, frames, done = 0.0, 0, False
        terminated, reward = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            frames += 1
            done = terminated or truncated
        scores.append(total)
        lengths.append(frames)
        outcomes.append(classify(terminated, reward))
    env.close()
    return {
        "eval_episodes": episodes,
        "eval_mean": round(float(np.mean(scores)), 1),
        "eval_std": round(float(np.std(scores)), 1),
        "eval_mean_frames": round(float(np.mean(lengths)), 1),
        "landed": outcomes.count("landed"),
        "crashed": outcomes.count("crashed"),
        "timed_out": outcomes.count("timed_out"),
        "scores": [round(s, 1) for s in scores],
    }
