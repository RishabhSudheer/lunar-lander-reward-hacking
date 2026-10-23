import json

from .config import RESULTS_DIR, run_tag


def results_path(variant, seed):
    return RESULTS_DIR / f"{run_tag(variant, seed)}.json"


def load(variant, seed):
    path = results_path(variant, seed)
    return json.loads(path.read_text()) if path.exists() else {}


def save(variant, seed, **fields):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load(variant, seed)
    data.update(variant=variant, seed=seed, **fields)
    results_path(variant, seed).write_text(json.dumps(data, indent=2))
    return data


def load_all():
    return [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*_s*.json"))]
