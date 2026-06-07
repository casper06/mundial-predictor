"""
Full knockout match prediction: 90min → Extra Time → Penalties.

For a knockout match, draws in 90min go to extra time.
Draws in extra time go to penalties.
"""
from dataclasses import dataclass
from .poisson import PoissonModel, expected_goals, match_probabilities
from .penalties import get_penalty_win_prob


@dataclass
class KnockoutPrediction:
    # Team names
    team_a: str
    team_b: str

    # 90 minutes
    p90_a:    float   # P(A wins in 90min)
    p90_draw: float   # P(draw after 90min)
    p90_b:    float   # P(B wins in 90min)

    # Extra time (conditional on reaching ET)
    pet_a:    float   # P(A wins in ET | ET reached)
    pet_draw: float   # P(draw after ET | ET reached)
    pet_b:    float   # P(B wins in ET | ET reached)

    # Penalties (conditional on reaching pens)
    ppen_a:   float   # P(A wins on pens | pens reached)
    ppen_b:   float   # P(B wins on pens | pens reached)

    # Overall: P(A/B wins the match by any route)
    p_a_wins: float
    p_b_wins: float

    # Probability of reaching each stage
    p_et:     float   # P(extra time played)
    p_pens:   float   # P(penalties played)

    # Expected goals (90min)
    xg_a: float
    xg_b: float


def predict_knockout(model: PoissonModel,
                     team_a: str, team_b: str) -> KnockoutPrediction:
    """
    Full knockout prediction for a neutral-ground match.
    """
    ra = model.get_rating(team_a)
    rb = model.get_rating(team_b)

    # ── 90 minutes ────────────────────────────────────────────────────────
    lam_a90, lam_b90 = expected_goals(ra, rb, home_advantage=0)
    p90_a, p90_draw, p90_b = match_probabilities(lam_a90, lam_b90)

    # ── Extra time (30 min = 1/3 of a game) ───────────────────────────────
    # Scale lambdas to 30 minutes
    lam_aET = lam_a90 * (30 / 90)
    lam_bET = lam_b90 * (30 / 90)
    pet_a, pet_draw, pet_b = match_probabilities(lam_aET, lam_bET)

    # ── Penalties ─────────────────────────────────────────────────────────
    ppen_a = get_penalty_win_prob(team_a, team_b)
    ppen_b = 1.0 - ppen_a

    # ── Overall probabilities ─────────────────────────────────────────────
    # P(reach ET) = P(draw after 90)
    p_et   = p90_draw

    # P(reach pens) = P(ET) * P(draw in ET)
    p_pens = p_et * pet_draw

    # P(A wins) = win in 90 + win in ET + win on pens
    p_a_wins = (
        p90_a
        + p_et   * pet_a
        + p_pens * ppen_a
    )
    p_b_wins = (
        p90_b
        + p_et   * pet_b
        + p_pens * ppen_b
    )

    return KnockoutPrediction(
        team_a   = team_a,
        team_b   = team_b,
        p90_a    = round(p90_a,    4),
        p90_draw = round(p90_draw, 4),
        p90_b    = round(p90_b,    4),
        pet_a    = round(pet_a,    4),
        pet_draw = round(pet_draw, 4),
        pet_b    = round(pet_b,    4),
        ppen_a   = round(ppen_a,   4),
        ppen_b   = round(ppen_b,   4),
        p_a_wins = round(p_a_wins, 4),
        p_b_wins = round(p_b_wins, 4),
        p_et     = round(p_et,     4),
        p_pens   = round(p_pens,   4),
        xg_a     = round(lam_a90,  2),
        xg_b     = round(lam_b90,  2),
    )