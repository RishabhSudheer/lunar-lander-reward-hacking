from pathlib import Path

ENV_ID = "LunarLander-v3"
SOLVED_SCORE = 200
MAX_FRAMES = 1000

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
VIDEOS_DIR = ROOT / "videos"

DEFAULT_STEPS = 1_000_000
DEFAULT_SEEDS = (0, 1)
N_ENVS = 16

PPO_KWARGS = dict(
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
)

EVAL_EPISODES = 20
EVAL_SEED_START = 1000

UPRIGHT_MAX_TILT = 0.2
SLOW_MAX_SPEED = 0.3
HOVER_BONUS = 1.0
FRAME_FEE = 0.05


def run_tag(variant, seed):
    return f"{variant}_s{seed}"
