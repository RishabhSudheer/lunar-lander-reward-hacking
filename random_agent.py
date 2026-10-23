import argparse

from gymnasium.wrappers import RecordVideo

from lander import make_env
from lander.config import VIDEOS_DIR

OBSERVATION_NAMES = [
    "x position",
    "y position",
    "x velocity",
    "y velocity",
    "angle",
    "angular velocity",
    "left leg contact",
    "right leg contact",
]


def main():
    parser = argparse.ArgumentParser(description="Fly the lander with random actions.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    env = make_env("baseline", render_mode="rgb_array" if args.record else None)
    if args.record:
        env = RecordVideo(env, video_folder=str(VIDEOS_DIR), name_prefix="random_agent",
                          episode_trigger=lambda _: True)

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=episode)
        if episode == 0:
            print("First observation:")
            for name, value in zip(OBSERVATION_NAMES, obs):
                print(f"  {name:18s} {value: .3f}")
        total, frames, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            frames += 1
            done = terminated or truncated
        outcome = "crashed" if reward == -100 else "landed" if reward == 100 else "timed out"
        print(f"Episode {episode}: {frames} frames, score {total:.1f}, {outcome}")
    env.close()


if __name__ == "__main__":
    main()
