import argparse

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from lander import VARIANTS, make_env
from lander.config import EVAL_SEED_START, MODELS_DIR, VIDEOS_DIR, run_tag


def main():
    parser = argparse.ArgumentParser(description="Record a trained model flying.")
    parser.add_argument("variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-seed", type=int, default=EVAL_SEED_START)
    args = parser.parse_args()

    tag = run_tag(args.variant, args.seed)
    model = PPO.load(MODELS_DIR / tag)
    env = RecordVideo(make_env("baseline", render_mode="rgb_array"),
                      video_folder=str(VIDEOS_DIR), name_prefix=tag,
                      episode_trigger=lambda _: True)

    obs, _ = env.reset(seed=args.episode_seed)
    frames, done = 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames += 1
        done = terminated or truncated
    env.close()
    print(f"[{tag}] recorded {frames} frames to {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
