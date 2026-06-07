"""
FastAPI backend for the World Cup 2026 predictor.

On startup it trains the Elo model once and keeps it in memory.
Then it serves predictions and rankings via JSON endpoints, and
serves the static web frontend.

Run from the project root:
    uvicorn api.main:app --reload

Then open http://localhost:8000
"""
from pathlib import Path
from pydantic import BaseModel as PydanticModel
from copy import deepcopy

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.data.loader import load_matches
from src.models.elo import EloModel
from src.models.poisson import PoissonModel
from src.models.knockout import predict_knockout
from src.models.penalties import get_penalty_win_prob
from src.simulation.monte_carlo import run_simulation, STAGE_ORDER

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

# ── World Cup 2026 groups (single source of truth) ──────────────────────────
GROUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
# Host nation bonus: applied only when a team plays IN its own country.
# Mexico is only "home" during the group stage (its venues); from the
# knockouts onward Mexico plays in the USA and loses the bonus.
HOST_BONUS = {
    "Mexico": 40,
    "United States": 25,
    "Canada": 15,
}

# Which country each host city belongs to
CITY_COUNTRY = {
    "Mexico City": "Mexico", "Zapopan": "Mexico", "Monterrey": "Mexico",
    "Toronto": "Canada", "Vancouver": "Canada",
    "Inglewood": "United States", "Santa Clara": "United States",
    "East Rutherford": "United States", "Foxborough": "United States",
    "Houston": "United States", "Arlington": "United States",
    "Philadelphia": "United States", "Atlanta": "United States",
    "Seattle": "United States", "Miami Gardens": "United States",
    "Kansas City": "United States",
}

def host_bonus(team: str, host_country: str | None) -> float:
    """Bonus only if the team is playing in its own country."""
    if host_country and HOST_BONUS.get(team) and CITY_COUNTRY.get(host_country) == team:
        return HOST_BONUS[team]
    # host_country may be passed as a country name directly
    if host_country and HOST_BONUS.get(team) and host_country == team:
        return HOST_BONUS[team]
    return 0.0

# Map team -> group letter for quick lookup
TEAM_GROUP = {team: g for g, teams in GROUPS.items() for team in teams}

# ── App setup ───────────────────────────────────────────────────────────────

class AdjustedModel:
    """
    Wraps the base model and applies temporary rating adjustments
    for specific teams. Does NOT modify the underlying model.
    Used for hypothetical scenarios (injuries, current form, etc.)
    """
    def __init__(self, base_model, adjustments: dict[str, float]):
        self.base = base_model
        self.adjustments = adjustments

    def get_rating(self, team: str) -> float:
        return (self.base.get_rating(team)
                + self.adjustments.get(team, 0.0)
                + HOST_BONUS.get(team, 0.0))

    @property
    def elo(self):
        return self.base.elo

    def predict_match(self, home, away, neutral=False):
        # Temporarily build prediction using adjusted ratings
        from src.models.poisson import expected_goals, match_probabilities
        from src.models.base import MatchPrediction
        rH = self.get_rating(home)
        rA = self.get_rating(away)
        bonus = 0.0 if neutral else self.base.elo.home_advantage
        lh, la = expected_goals(rH, rA, home_advantage=bonus)
        ph, pd, pa = match_probabilities(lh, la)
        return MatchPrediction(round(ph,4), round(pd,4), round(pa,4), round(lh,2), round(la,2))

# ── Request models for POST endpoints ────────────────────────────────────────
class ExtraResult(PydanticModel):
    home: str
    away: str
    home_goals: int
    away_goals: int

class PredictRequest(PydanticModel):
    home: str
    away: str
    knockout: bool = False
    home_adjust: float = 0.0
    away_adjust: float = 0.0
    extra_results: list[ExtraResult] = []
    use_extra: bool = False


def build_model_with_extra(extra_results: list[ExtraResult]) -> PoissonModel:
    """
    Clone the base Elo model and apply extra (real tournament) results
    on top, then wrap in Poisson. Does not touch the base model.
    """
    base_elo = STATE["elo"]
    elo = deepcopy(base_elo)
    for r in extra_results:
        elo.update(
            home=r.home, away=r.away,
            home_goals=r.home_goals, away_goals=r.away_goals,
            tournament="FIFA World Cup",   # real WC matches get full weight
            neutral=True,
        )
    return PoissonModel(elo)

app = FastAPI(title="World Cup 2026 Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model is trained once on startup and stored here
STATE: dict = {}


@app.on_event("startup")
def train_model():
    print("Training Elo model on full dataset...")
    df = load_matches()
    elo = EloModel()
    for _, row in df.iterrows():
        elo.update(
            home=row["home_team"], away=row["away_team"],
            home_goals=row["home_score"], away_goals=row["away_score"],
            tournament=row["tournament"], neutral=row["neutral"],
        )
    STATE["elo"] = elo
    STATE["poisson"] = PoissonModel(elo)
    print(f"Model trained on {len(df):,} matches.")
    print("Running Monte Carlo simulation (10,000 tournaments)...")
    STATE["simulation"] = run_simulation(STATE["poisson"], n=10_000)
    print("Simulation complete. Server ready.")


# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

@app.get("/api/rankings")
def rankings():
    """Return all WC2026 teams sorted by Elo rating."""
    elo = STATE["elo"]
    teams = []
    for team, group in TEAM_GROUP.items():
        teams.append({
            "team": team,
            "group": group,
            "rating": round(elo.get_rating(team), 1),
        })
    teams.sort(key=lambda t: t["rating"], reverse=True)
    for i, t in enumerate(teams, 1):
        t["rank"] = i
    return {"teams": teams}


@app.get("/api/predict")
def predict(
    home: str = Query(...),
    away: str = Query(...),
    knockout: bool = Query(False),
    home_adjust: float = Query(0.0),   # rating adjustment for home team
    away_adjust: float = Query(0.0),   # rating adjustment for away team
):
    """
    Predict a single match. Neutral ground assumed (World Cup).
    home_adjust / away_adjust: temporary rating deltas (injuries, form, etc.)
    that apply ONLY to this prediction, without changing the base model.
    """
    base = STATE["poisson"]
    model = AdjustedModel(base, {home: home_adjust, away: away_adjust})

    if knockout:
        k = predict_knockout(model, home, away)
        return {
            "type": "knockout",
            "home": home, "away": away,
            "p90": {"home": k.p90_a, "draw": k.p90_draw, "away": k.p90_b},
            "extra_time": {"home": k.pet_a, "draw": k.pet_draw, "away": k.pet_b,
                           "reached": k.p_et},
            "penalties": {"home": k.ppen_a, "away": k.ppen_b, "reached": k.p_pens},
            "overall": {"home": k.p_a_wins, "away": k.p_b_wins},
            "xg": {"home": k.xg_a, "away": k.xg_b},
            "scorelines": _scorelines_per_outcome(model, home, away),
            "adjustments": {"home": home_adjust, "away": away_adjust},
        }
    else:
        p = model.predict_match(home, away, neutral=True)
        return {
            "type": "group",
            "home": home, "away": away,
            "probs": {"home": p.p_home_win, "draw": p.p_draw, "away": p.p_away_win},
            "xg": {"home": p.exp_home_goals, "away": p.exp_away_goals},
            "scorelines": _scorelines_per_outcome(model, home, away),
            "adjustments": {"home": home_adjust, "away": away_adjust},
        }


def _scorelines_per_outcome(model, home, away):
    """Best scoreline for each outcome (home win / draw / away win)."""
    import math
    from src.models.poisson import expected_goals, poisson_prob, MAX_GOALS
    rH = model.get_rating(home)
    rA = model.get_rating(away)
    lh, la = expected_goals(rH, rA, home_advantage=0)

    best = {"home": ("", -1), "draw": ("", -1), "away": ("", -1)}
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson_prob(lh, i) * poisson_prob(la, j)
            if i > j and p > best["home"][1]:
                best["home"] = (f"{i}-{j}", p)
            elif i == j and p > best["draw"][1]:
                best["draw"] = (f"{i}-{j}", p)
            elif i < j and p > best["away"][1]:
                best["away"] = (f"{i}-{j}", p)

    return {
        "home": {"score": best["home"][0], "prob": round(best["home"][1], 4)},
        "draw": {"score": best["draw"][0], "prob": round(best["draw"][1], 4)},
        "away": {"score": best["away"][0], "prob": round(best["away"][1], 4)},
    }


@app.get("/api/groups")
def groups():
    """Return the WC2026 group assignments."""
    return {"groups": GROUPS}

@app.get("/api/simulation")
def simulation():
    """Return tournament simulation: P(each team reaches each stage)."""
    sim = STATE["simulation"]
    teams = []
    for team, stages in sim.items():
        if team not in TEAM_GROUP:
            continue
        teams.append({
            "team": team,
            "group": TEAM_GROUP[team],
            "champion": round(stages.get("Champion", 0) * 100, 1),
            "final": round(stages.get("Final", 0) * 100, 1),
            "semi": round(stages.get("SF", 0) * 100, 1),
            "quarter": round(stages.get("QF", 0) * 100, 1),
            "r16": round(stages.get("R16", 0) * 100, 1),
            "r32": round(stages.get("R32", 0) * 100, 1),
        })
    teams.sort(key=lambda t: t["champion"], reverse=True)
    return {"teams": teams}

@app.post("/api/predict")
def predict_post(req: PredictRequest):
    """
    Predict a match, optionally applying extra real results to the model first.
    Used by the Fixture feature: when use_extra is true, the model is
    retrained with the tournament results the user has entered.
    """
    if req.use_extra and req.extra_results:
        base = build_model_with_extra(req.extra_results)
    else:
        base = STATE["poisson"]

    model = AdjustedModel(base, {req.home: req.home_adjust, req.away: req.away_adjust})

    if req.knockout:
        k = predict_knockout(model, req.home, req.away)
        return {
            "type": "knockout", "home": req.home, "away": req.away,
            "p90": {"home": k.p90_a, "draw": k.p90_draw, "away": k.p90_b},
            "extra_time": {"home": k.pet_a, "draw": k.pet_draw, "away": k.pet_b, "reached": k.p_et},
            "penalties": {"home": k.ppen_a, "away": k.ppen_b, "reached": k.p_pens},
            "overall": {"home": k.p_a_wins, "away": k.p_b_wins},
            "xg": {"home": k.xg_a, "away": k.xg_b},
            "scorelines": _scorelines_per_outcome(model, req.home, req.away),
        }
    else:
        p = model.predict_match(req.home, req.away, neutral=True)
        return {
            "type": "group", "home": req.home, "away": req.away,
            "probs": {"home": p.p_home_win, "draw": p.p_draw, "away": p.p_away_win},
            "xg": {"home": p.exp_home_goals, "away": p.exp_away_goals},
            "scorelines": _scorelines_per_outcome(model, req.home, req.away),
        }

# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")

# Static files (CSS, JS, images) under /static
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")