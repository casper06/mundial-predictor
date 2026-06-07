"""
Penalty shootout win probability based on historical shootout data.

Uses shootouts.csv from the Kaggle dataset to build a win rate per team.
Falls back to 0.50 for teams with no shootout history.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHOOTOUTS_CSV = ROOT / "data" / "raw" / "shootouts.csv"

# Minimum shootouts needed to trust the historical rate.
# Below this, we blend with the 0.50 prior.
MIN_SHOOTOUTS = 3


def build_penalty_ratings(path: Path = SHOOTOUTS_CSV) -> dict[str, float]:
    if not path.exists():
        return {}

    # Import NAME_MAP from loader to normalise team names
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.loader import NAME_MAP

    df = pd.read_csv(path)
    df["home_team"] = df["home_team"].str.strip().replace(NAME_MAP)
    df["away_team"] = df["away_team"].str.strip().replace(NAME_MAP)
    df["winner"]    = df["winner"].str.strip().replace(NAME_MAP)

    ratings = {}
    all_teams = set(df["home_team"]) | set(df["away_team"])

    for team in all_teams:
        played = df[(df["home_team"] == team) | (df["away_team"] == team)]
        won    = played[played["winner"] == team]
        n, w   = len(played), len(won)
        alpha  = min(n / MIN_SHOOTOUTS, 1.0)
        rate   = w / n if n > 0 else 0.5
        ratings[team] = alpha * rate + (1 - alpha) * 0.50

    return ratings

# Singleton — load once
_RATINGS: dict[str, float] | None = None

def get_penalty_win_prob(team_a: str, team_b: str) -> float:
    """
    P(team_a wins the shootout against team_b).
    Uses relative strength of both teams, not just one.
    """
    global _RATINGS
    if _RATINGS is None:
        _RATINGS = build_penalty_ratings()

    pa = _RATINGS.get(team_a, 0.50)
    pb = _RATINGS.get(team_b, 0.50)

    # Normalize so they sum to 1
    return pa / (pa + pb)


if __name__ == "__main__":
    ratings = build_penalty_ratings()
    ranked  = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Team':<30} {'Penalty Win %':>14}")
    print("-" * 46)
    for team, rate in ranked:
        print(f"{team:<30} {rate:>13.1%}")