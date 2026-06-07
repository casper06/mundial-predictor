"""
Test knockout predictions for famous potential matchups.
Run: python predict_knockout.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_matches
from src.models.elo import EloModel
from src.models.poisson import PoissonModel
from src.models.knockout import predict_knockout

def train():
    df = load_matches()
    elo = EloModel()
    for _, row in df.iterrows():
        elo.update(row["home_team"], row["away_team"],
                   row["home_score"], row["away_score"],
                   row["tournament"], row["neutral"])
    return PoissonModel(elo)

def print_knockout(pred):
    a, b = pred.team_a, pred.team_b
    print(f"\n{'='*55}")
    print(f"  {a}  vs  {b}")
    print(f"  (knockout, neutral ground)")
    print(f"{'='*55}")
    print(f"\n  90 MINUTES")
    print(f"    {a:<22} {pred.p90_a:>6.1%}")
    print(f"    Draw (→ ET)        {pred.p90_draw:>6.1%}")
    print(f"    {b:<22} {pred.p90_b:>6.1%}")
    print(f"    xG: {pred.xg_a:.2f} — {pred.xg_b:.2f}")

    print(f"\n  EXTRA TIME  (P reaching ET: {pred.p_et:.1%})")
    print(f"    {a:<22} {pred.pet_a:>6.1%}")
    print(f"    Draw (→ pens)      {pred.pet_draw:>6.1%}")
    print(f"    {b:<22} {pred.pet_b:>6.1%}")

    print(f"\n  PENALTIES   (P reaching pens: {pred.p_pens:.1%})")
    print(f"    {a:<22} {pred.ppen_a:>6.1%}")
    print(f"    {b:<22} {pred.ppen_b:>6.1%}")

    print(f"\n  OVERALL WINNER")
    print(f"    {a:<22} {pred.p_a_wins:>6.1%}")
    print(f"    {b:<22} {pred.p_b_wins:>6.1%}")

if __name__ == "__main__":
    print("Training model...")
    model = train()

    matchups = [
        ("Argentina", "France"),
        ("Germany",   "England"),
        ("Brazil",    "Netherlands"),
        ("Spain",     "Portugal"),
    ]

    for a, b in matchups:
        print_knockout(predict_knockout(model, a, b))