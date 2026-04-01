import json
from pathlib import Path
from typing import TypedDict


class VoteStore(TypedDict):
    unet: int
    felix: int


def _default_votes() -> VoteStore:
    return {"unet": 0, "felix": 0}


def load_votes(filepath: Path) -> VoteStore:
    if not filepath.exists():
        save_votes(filepath, _default_votes())
        return _default_votes()

    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "unet": int(data.get("unet", 0)),
            "felix": int(data.get("felix", 0)),
        }
    except (json.JSONDecodeError, OSError, ValueError):
        save_votes(filepath, _default_votes())
        return _default_votes()


def save_votes(filepath: Path, votes: VoteStore) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(votes, f, ensure_ascii=False, indent=2)


def register_vote(filepath: Path, model_name: str) -> VoteStore:
    if model_name not in {"unet", "felix"}:
        raise ValueError("model must be 'unet' or 'felix'")

    votes = load_votes(filepath)
    votes[model_name] += 1
    save_votes(filepath, votes)
    return votes


def compute_vote_stats(votes: VoteStore) -> dict[str, float | int]:
    total = votes["unet"] + votes["felix"]
    if total == 0:
        return {
            "unet_votes": votes["unet"],
            "felix_votes": votes["felix"],
            "total_votes": 0,
            "unet_percentage": 0.0,
            "felix_percentage": 0.0,
        }

    return {
        "unet_votes": votes["unet"],
        "felix_votes": votes["felix"],
        "total_votes": total,
        "unet_percentage": round((votes["unet"] / total) * 100, 2),
        "felix_percentage": round((votes["felix"] / total) * 100, 2),
    }
