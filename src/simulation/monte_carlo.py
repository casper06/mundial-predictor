"""
Monte Carlo simulation of the 2026 FIFA World Cup.

Simulates the full tournament N times and returns the probability
of each team reaching each stage (champion, final, semi, quarters, R32, R16).

Format:
  - 12 groups of 4 teams
  - Top 2 from each group + 8 best third-place teams -> Round of 32
  - Single elimination: R32 -> R16 -> QF -> SF -> Final

Host-nation bonus is applied per match using the REAL venue of each match
(from the official 2026 calendar), so e.g. Mexico is only "home" in its
group matches and the few R32/R16 matches played in Mexico; the USA is
home in the 26 knockout matches played on US soil, including the final.
"""
import random
from collections import defaultdict
from dataclasses import dataclass

from src.models.poisson import PoissonModel, poisson_prob, expected_goals

# ---------------------------------------------------------------------------
# Host bonus
# ---------------------------------------------------------------------------
HOST_BONUS = {"Mexico": 40, "United States": 25, "Canada": 15}

# ---------------------------------------------------------------------------
# Tournament data
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
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

HOSTS = {"Mexico", "United States", "Canada"}

# Round of 32: each match has (match_no, slot_home, slot_away).
# Slots: "A1"=winner group A, "A2"=runner-up A, "T1".."T8"=best third places.
R32_MATCHES = [
    (73, "A2", "B2"), (76, "C1", "F2"), (74, "E1", "T1"), (75, "F1", "C2"),
    (78, "E2", "I2"), (77, "I1", "T2"), (79, "A1", "T3"), (80, "L1", "T4"),
    (82, "G1", "T5"), (81, "D1", "T6"), (84, "H1", "J2"), (83, "K2", "L2"),
    (85, "B1", "T7"), (88, "D2", "G2"), (86, "J1", "H2"), (87, "K1", "T8"),
]

# Round of 16: (match_no, feeder_match_a, feeder_match_b)
R16_MATCHES = [
    (90, 73, 75), (89, 74, 77), (91, 76, 78), (92, 79, 80),
    (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87),
]
QF_MATCHES = [
    (97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96),
]
SF_MATCHES = [
    (101, 97, 98), (102, 99, 100),
]
FINAL_MATCH = (104, 101, 102)

# Host country of each knockout match by match number (from official calendar)
KO_VENUES = {
    73:"United States",76:"United States",74:"United States",75:"Mexico",
    78:"United States",77:"United States",79:"Mexico",80:"United States",
    82:"United States",81:"United States",84:"United States",83:"Canada",
    85:"Canada",88:"United States",86:"United States",87:"United States",
    90:"United States",89:"United States",91:"United States",92:"Mexico",
    93:"United States",94:"United States",95:"United States",96:"Canada",
    97:"United States",98:"United States",99:"United States",100:"United States",
    101:"United States",102:"United States",104:"United States",
}

# ---------------------------------------------------------------------------
# Match simulation
# ---------------------------------------------------------------------------
def _poisson_sample(lam: float) -> int:
    l = pow(2.718281828, -lam)
    k, p = 0, 1.0
    while p > l:
        k += 1
        p *= random.random()
    return k - 1


def simulate_match(model: PoissonModel, home: str, away: str,
                   knockout: bool = False, host_country: str | None = None
                   ) -> tuple[str, int, int]:
    """
    Simulate one match. host_country: the nation hosting this specific match;
    a host team playing in its own country gets a rating bonus.
    """
    r_home = model.get_rating(home)
    r_away = model.get_rating(away)

    if host_country:
        if home == host_country:
            r_home += HOST_BONUS.get(home, 0)
        if away == host_country:
            r_away += HOST_BONUS.get(away, 0)

    lam_h, lam_a = expected_goals(r_home, r_away)
    hg = _poisson_sample(lam_h)
    ag = _poisson_sample(lam_a)

    if knockout and hg == ag:
        p = model.predict_match(home, away, neutral=True)
        denom = p.p_home_win + p.p_away_win
        winner = home if random.random() < (p.p_home_win / denom if denom else 0.5) else away
        return winner, hg, ag

    winner = home if hg > ag else (away if ag > hg else "draw")
    return winner, hg, ag


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------
@dataclass
class TeamRecord:
    team: str
    pts: int = 0
    gd: int = 0
    gf: int = 0
    group: str = ""


def simulate_group(model: PoissonModel, group_name: str,
                   teams: list[str]) -> list[TeamRecord]:
    records = {t: TeamRecord(team=t, group=group_name) for t in teams}
    # The host nation in this group plays its group matches at home
    group_host = next((t for t in teams if t in HOSTS), None)

    for i, home in enumerate(teams):
        for away in teams[i+1:]:
            winner, hg, ag = simulate_match(model, home, away,
                                            host_country=group_host)
            records[home].gf += hg
            records[away].gf += ag
            records[home].gd += hg - ag
            records[away].gd += ag - hg
            if winner == home:
                records[home].pts += 3
            elif winner == away:
                records[away].pts += 3
            else:
                records[home].pts += 1
                records[away].pts += 1

    return sorted(records.values(),
                  key=lambda r: (r.pts, r.gd, r.gf, random.random()),
                  reverse=True)


# ---------------------------------------------------------------------------
# Full tournament
# ---------------------------------------------------------------------------
def simulate_tournament(model: PoissonModel) -> dict[str, str]:
    stages: dict[str, str] = {}
    group_results: dict[str, list[TeamRecord]] = {}
    all_thirds: list[TeamRecord] = []

    for g, teams in GROUPS.items():
        standing = simulate_group(model, g, teams)
        group_results[g] = standing
        first, second, third, fourth = standing
        stages[first.team] = "R32"
        stages[second.team] = "R32"
        stages[third.team] = "group_third"
        stages[fourth.team] = "group_stage"
        all_thirds.append(third)

    best_thirds = sorted(all_thirds,
                         key=lambda r: (r.pts, r.gd, r.gf, random.random()),
                         reverse=True)[:8]
    for t in best_thirds:
        stages[t.team] = "R32"

    # Build slots
    slots: dict[str, str] = {}
    for g, standing in group_results.items():
        slots[f"{g}1"] = standing[0].team
        slots[f"{g}2"] = standing[1].team
    for i, t in enumerate(best_thirds, 1):
        slots[f"T{i}"] = t.team

    # winners[match_no] = winning team of that match
    winners: dict[int, str] = {}

    def play(match_no, home, away, round_name):
        host = KO_VENUES.get(match_no)
        winner, _, _ = simulate_match(model, home, away,
                                      knockout=True, host_country=host)
        loser = away if winner == home else home
        if stages.get(loser) in ("R32", "group_third"):
            stages[loser] = round_name
        winners[match_no] = winner
        return winner

    # R32
    for mn, sh, sa in R32_MATCHES:
        play(mn, slots[sh], slots[sa], "R32")
    # R16
    for mn, fa, fb in R16_MATCHES:
        play(mn, winners[fa], winners[fb], "R16")
    # QF
    for mn, fa, fb in QF_MATCHES:
        play(mn, winners[fa], winners[fb], "QF")
    # SF
    for mn, fa, fb in SF_MATCHES:
        play(mn, winners[fa], winners[fb], "SF")
    # Final
    mn, fa, fb = FINAL_MATCH
    champ = play(mn, winners[fa], winners[fb], "Final")
    stages[champ] = "Champion"

    return stages


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------
STAGE_ORDER = ["group_stage", "group_third", "R32", "R16", "QF", "SF", "Final", "Champion"]


def run_simulation(model: PoissonModel, n: int = 10_000) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _ in range(n):
        result = simulate_tournament(model)
        for team, stage in result.items():
            idx = STAGE_ORDER.index(stage)
            for s in STAGE_ORDER[:idx+1]:
                counts[team][s] += 1
    probs = {}
    for team, sc in counts.items():
        probs[team] = {stage: sc[stage] / n for stage in STAGE_ORDER}
    return probs
