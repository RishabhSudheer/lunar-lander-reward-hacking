import gymnasium as gym
import numpy as np

from .config import (
    ENV_ID,
    FRAME_FEE,
    HOVER_BONUS,
    SLOW_MAX_SPEED,
    UPRIGHT_MAX_TILT,
)


def upright_and_slow(obs):
    vx, vy, tilt = obs[2], obs[3], obs[4]
    return abs(tilt) < UPRIGHT_MAX_TILT and np.hypot(vx, vy) < SLOW_MAX_SPEED


class TimePenalty(gym.Wrapper):
    def __init__(self, env, fee=FRAME_FEE):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info


class HoverBonus(gym.Wrapper):
    def __init__(self, env, bonus=HOVER_BONUS):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if upright_and_slow(obs):
            reward += self.bonus
        return obs, reward, terminated, truncated, info


class ProgressBonus(gym.Wrapper):
    def __init__(self, env, bonus=HOVER_BONUS, fee=FRAME_FEE):
        super().__init__(env)
        self.bonus = bonus
        self.fee = fee
        self._previous = 0.0

    def _score(self, obs):
        return self.bonus * float(upright_and_slow(obs))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._previous = self._score(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        score = self._score(obs)
        reward += (score - self._previous) - self.fee
        self._previous = score
        return obs, reward, terminated, truncated, info


VARIANTS = {
    "baseline": None,
    "time_penalty": TimePenalty,
    "hover_hack": HoverBonus,
    "hover_fixed": ProgressBonus,
}


def make_env(variant="baseline", render_mode=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {', '.join(VARIANTS)}")
    env = gym.make(ENV_ID, render_mode=render_mode)
    wrapper = VARIANTS[variant]
    return wrapper(env) if wrapper else env
