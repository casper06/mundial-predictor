"""
Poisson goal model built on top of Elo ratings.

The Elo model tells us HOW STRONG each team is.
The Poisson model converts that into:
  - Expected goals for each team
  - Real draw probability (not an approximation)
  - Full scoreline probabilities (e.g. P(2-1) = X%)

How it works:
  Goals in football follow a Poisson distribution reasonably well.
  If we expect Argentina to score 1.8 goals, the probability of them
  scoring exactly 2 is: e^(-1.8) * 1.8^2 / 2! = 26.8%

  We derive expected goals from the Elo rating difference using a
  formula calibrated on historical data.
"""
import math
from .base import BaseModel, MatchPrediction
from .elo import EloModel, SCALE

# Average goals per team per match in international football (historical avg)
BASE_GOALS = 1.35

# How much the rating difference affects expected goals.
# Calibrated so that a 200-point difference ~ 0.3 extra goals per game.
GOAL_SENSITIVITY = 0.0015

# Maximum scoreline to consider (8-8 covers 99.9%+ of real matches)
MAX_GOALS = 8


def expected_goals(rating_a: float, rating_b: float,
                   home_advantage: float = 0.0) -> tuple[float, float]:
    """
    Convert Elo ratings into expected goals for each team.

    Args:
        rating_a: Elo rating of team A (the "home" side for this calc)
        rating_b: Elo rating of team B
        home_advantage: extra rating points for playing at home (0 if neutral)

    Returns:
        (lambda_a, lambda_b) — Poisson lambdas (expected goals)
    """
    diff = (rating_a + home_advantage) - rating_b
    lambda_a = BASE_GOALS * math.exp( GOAL_SENSITIVITY * diff)
    lambda_b = BASE_GOALS * math.exp(-GOAL_SENSITIVITY * diff)
    return lambda_a, lambda_b


def poisson_prob(lam: float, k: int) -> float:
    """P(X = k) where X ~ Poisson(lambda)."""
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def match_probabilities(lambda_home: float, lambda_away: float
                        ) -> tuple[float, float, float]:
    """
    Compute P(home win), P(draw), P(away win) from Poisson lambdas.

    We build the full scoreline matrix up to MAX_GOALS x MAX_GOALS
    and sum the probabilities in each region.
    """
    p_home = p_draw = p_away = 0.0

    for i in range(MAX_GOALS + 1):       # home goals
        for j in range(MAX_GOALS + 1):   # away goals
            p = poisson_prob(lambda_home, i) * poisson_prob(lambda_away, j)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    # Normalise to 1.0 (tiny rounding error from truncating at MAX_GOALS)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


class PoissonModel(BaseModel):
    """
    Wraps an EloModel and adds Poisson-based goal prediction.

    The Elo model is still the source of truth for team strength.
    Poisson just converts ratings → goals → probabilities more accurately.
    """

    def __init__(self, elo_model: EloModel):
        self.elo = elo_model

    def get_rating(self, team: str) -> float:
        return self.elo.get_rating(team)

    def predict_match(self, home: str, away: str,
                      neutral: bool = False) -> MatchPrediction:
        r_home = self.elo.get_rating(home)
        r_away = self.elo.get_rating(away)
        bonus  = 0.0 if neutral else self.elo.home_advantage

        lam_h, lam_a = expected_goals(r_home, r_away, home_advantage=bonus)
        p_h, p_d, p_a = match_probabilities(lam_h, lam_a)

        return MatchPrediction(
            p_home_win      = round(p_h, 4),
            p_draw          = round(p_d, 4),
            p_away_win      = round(p_a, 4),
            exp_home_goals  = round(lam_h, 2),
            exp_away_goals  = round(lam_a, 2),
        )

    def top_scorelines(self, home: str, away: str,
                       neutral: bool = False, top_n: int = 5) -> list[dict]:
        """
        Return the most likely exact scorelines for a match.

        Example output:
            [{"score": "1-1", "probability": 0.112}, ...]
        """
        r_home = self.elo.get_rating(home)
        r_away = self.elo.get_rating(away)
        bonus  = 0.0 if neutral else self.elo.home_advantage

        lam_h, lam_a = expected_goals(r_home, r_away, home_advantage=bonus)

        scores = []
        for i in range(MAX_GOALS + 1):
            for j in range(MAX_GOALS + 1):
                p = poisson_prob(lam_h, i) * poisson_prob(lam_a, j)
                scores.append({"score": f"{i}-{j}", "probability": p})

        scores.sort(key=lambda x: x["probability"], reverse=True)
        return scores[:top_n]