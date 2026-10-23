import argparse

from stable_baselines3 import PPO

from lander import VARIANTS
from lander import results
from lander.config import EVAL_EPISODES, MODELS_DIR, run_tag
from lander.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Test a trained model on the unmodified reward (wrapper off).")
    parser.add_argument("variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    tag = run_tag(args.variant, args.seed)
    model = PPO.load(MODELS_DIR / tag)
    summary = evaluate(model, episodes=args.episodes)
    results.save(args.variant, args.seed, **summary)

    print(f"[{tag}] {args.episodes} episodes, wrapper off: "
          f"mean {summary['eval_mean']} ± {summary['eval_std']}, "
          f"landed {summary['landed']}, crashed {summary['crashed']}, "
          f"timed out {summary['timed_out']}")


if __name__ == "__main__":
    main()
