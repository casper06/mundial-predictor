"""
Train the Elo model on the full historical dataset.

Run from the project root:
    python train_elo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_matches
from src.models.elo import EloModel

def train(cutoff_date: str | None = None) -> EloModel:
    df = load_matches()
    if cutoff_date:
        df = df[df["date"] < cutoff_date]
        print(f"Training on matches before {cutoff_date}: {len(df):,} matches")
    else:
        print(f"Training on full dataset: {len(df):,} matches")

    model = EloModel()
    for _, row in df.iterrows():
        model.update(
            home       = row["home_team"],
            away       = row["away_team"],
            home_goals = row["home_score"],
            away_goals = row["away_score"],
            tournament = row["tournament"],
            neutral    = row["neutral"],
        )
    return model

def print_ranking(model: EloModel, top_n: int = 30) -> None:
    ranking = sorted(model.ratings.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n{'Rank':<5} {'Team':<25} {'Rating':>7}")
    print("-" * 40)
    for i, (team, rating) in enumerate(ranking[:top_n], 1):
        print(f"{i:<5} {team:<25} {rating:>7.1f}")

if __name__ == "__main__":
    model = train()
    print_ranking(model, top_n=30)

    print("\n--- Sample predictions (neutral ground) ---")
    pairs = [
        ("Argentina", "France"),
        ("Brazil",    "Germany"),
        ("Spain",     "England"),
        ("Argentina", "Brazil"),
    ]
    print(f"\n{'Match':<35} {'Home win':>9} {'Draw':>7} {'Away win':>9}")
    print("-" * 65)
    for home, away in pairs:
        p = model.predict_match(home, away, neutral=True)
        print(f"{home} vs {away:<25} {p.p_home_win:>8.1%} {p.p_draw:>6.1%} {p.p_away_win:>8.1%}")