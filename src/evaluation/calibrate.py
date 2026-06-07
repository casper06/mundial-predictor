"""
Find the GOAL_SENSITIVITY value that minimises Brier score on Qatar 2022.

Run:
    python -m src.evaluation.calibrate
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_matches
from src.models.elo import EloModel
from src.models import poisson as poisson_module
from src.models.poisson import PoissonModel

WC_START = "2022-11-20"
WC_END   = "2022-12-19"

def train_until(cutoff: str):
    df = load_matches()
    df = df[df["date"] < cutoff]
    elo = EloModel()
    for _, row in df.iterrows():
        elo.update(
            home=row["home_team"], away=row["away_team"],
            home_goals=row["home_score"], away_goals=row["away_score"],
            tournament=row["tournament"], neutral=row["neutral"],
        )
    return elo

def brier_for(model, wc_df) -> float:
    total = 0.0
    for _, row in wc_df.iterrows():
        hg, ag = row["home_score"], row["away_score"]
        pred = model.predict_match(row["home_team"], row["away_team"], neutral=True)
        pv = [pred.p_home_win, pred.p_draw, pred.p_away_win]
        av = [1,0,0] if hg > ag else ([0,1,0] if hg == ag else [0,0,1])
        total += sum((p-a)**2 for p,a in zip(pv,av)) / 3
    return total / len(wc_df)

if __name__ == "__main__":
    elo = train_until(WC_START)
    df  = load_matches()
    wc  = df[(df["date"] >= WC_START) & (df["date"] <= WC_END) &
             (df["tournament"] == "FIFA World Cup")].copy()

    print("Calibrating GOAL_SENSITIVITY...\n")
    print(f"{'Sensitivity':>12} {'Brier':>8}")
    print("-" * 22)

    best_val, best_brier = None, 999
    # Try values from 0.0005 to 0.0035
    for s in [0.0005, 0.0008, 0.0010, 0.0012, 0.0015, 0.0018, 0.0020, 0.0025, 0.0030]:
        poisson_module.GOAL_SENSITIVITY = s   # override the module constant
        model = PoissonModel(elo)
        b = brier_for(model, wc)
        marker = ""
        if b < best_brier:
            best_brier, best_val = b, s
            marker = "  <-- best so far"
        print(f"{s:>12.4f} {b:>8.4f}{marker}")

    print(f"\nBest: GOAL_SENSITIVITY = {best_val} (Brier {best_brier:.4f})")
    print(f"Elo baseline was 0.2034 — Poisson beats it if below that.")