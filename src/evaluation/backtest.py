"""
Backtest Elo and Poisson models against Qatar 2022.

Run:
    python -m src.evaluation.backtest
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_matches
from src.models.elo import EloModel
from src.models.poisson import PoissonModel

WC_START = "2022-11-20"
WC_END   = "2022-12-19"

def train_until(cutoff: str):
    df = load_matches()
    df = df[df["date"] < cutoff]
    elo = EloModel()
    for _, row in df.iterrows():
        elo.update(
            home       = row["home_team"],
            away       = row["away_team"],
            home_goals = row["home_score"],
            away_goals = row["away_score"],
            tournament = row["tournament"],
            neutral    = row["neutral"],
        )
    return elo

def evaluate(model, wc_df) -> tuple[int, float]:
    correct = 0
    brier   = 0.0
    for _, row in wc_df.iterrows():
        hg, ag = row["home_score"], row["away_score"]
        pred   = model.predict_match(row["home_team"], row["away_team"], neutral=True)
        pv     = [pred.p_home_win, pred.p_draw, pred.p_away_win]
        av     = [1,0,0] if hg > ag else ([0,1,0] if hg == ag else [0,0,1])
        if pv.index(max(pv)) == av.index(1):
            correct += 1
        brier += sum((p-a)**2 for p,a in zip(pv,av)) / 3
    return correct, brier / len(wc_df)

if __name__ == "__main__":
    print("Training on all matches before Qatar 2022...")
    elo     = train_until(WC_START)
    poisson = PoissonModel(elo)

    df = load_matches()
    wc = df[
        (df["date"] >= WC_START) &
        (df["date"] <= WC_END) &
        (df["tournament"] == "FIFA World Cup")
    ].copy()

    print(f"Evaluating on {len(wc)} matches\n")

    elo_ok,  elo_brier  = evaluate(elo,     wc)
    poi_ok,  poi_brier  = evaluate(poisson, wc)
    n = len(wc)

    print(f"{'Model':<10} {'Accuracy':>10} {'Brier':>8}")
    print("-" * 32)
    print(f"{'Elo':<10} {elo_ok}/{n} = {elo_ok/n:.1%}  {elo_brier:.4f}")
    print(f"{'Poisson':<10} {poi_ok}/{n} = {poi_ok/n:.1%}  {poi_brier:.4f}")

    # Scoreline examples
    print("\n--- Expected scorelines (neutral ground) ---")
    pairs = [("Argentina","France"), ("Brazil","Germany"), ("Spain","England")]
    for home, away in pairs:
        pred = poisson.predict_match(home, away, neutral=True)
        tops = poisson.top_scorelines(home, away, neutral=True)
        print(f"\n{home} vs {away}")
        print(f"  Win {pred.p_home_win:.1%} | Draw {pred.p_draw:.1%} | Loss {pred.p_away_win:.1%}")
        print(f"  Expected goals: {pred.exp_home_goals:.2f} - {pred.exp_away_goals:.2f}")
        print(f"  Top scorelines: " + ", ".join(f"{s['score']}({s['probability']:.1%})" for s in tops))