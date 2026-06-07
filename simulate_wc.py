"""
Simulate the 2026 World Cup and print championship odds.

Run:
    python simulate_wc.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_matches
from src.models.elo import EloModel
from src.models.poisson import PoissonModel
from src.simulation.monte_carlo import run_simulation, STAGE_ORDER

def train_full() -> PoissonModel:
    df = load_matches()
    elo = EloModel()
    for _, row in df.iterrows():
        elo.update(
            home=row["home_team"], away=row["away_team"],
            home_goals=row["home_score"], away_goals=row["away_score"],
            tournament=row["tournament"], neutral=row["neutral"],
        )
    return PoissonModel(elo)

if __name__ == "__main__":
    print("Training model on full dataset...")
    model = train_full()

    print("Running 10,000 simulations of World Cup 2026...\n")
    probs = run_simulation(model, n=10_000)

    # Sort by champion probability
    ranked = sorted(probs.items(), key=lambda kv: kv[1]["Champion"], reverse=True)

    print(f"{'Team':<25} {'Champion':>9} {'Final':>7} {'Semi':>7} {'QF':>7} {'R16':>7} {'R32':>7}")
    print("-" * 75)
    for team, p in ranked:
        if p["R32"] > 0.01:   # only show teams that realistically make R32
            print(f"{team:<25} {p['Champion']:>8.1%} {p['Final']:>6.1%} "
                  f"{p['SF']:>6.1%} {p['QF']:>6.1%} {p['R16']:>6.1%} {p['R32']:>6.1%}")