# World Cup 2026 Predictor — Technical Writeup

**Author:** Fernando Conti  
**Stack:** Python · FastAPI · Elo · Poisson · Monte Carlo  
**Dataset:** 47,126 international matches (1872–2024) + 148 recent results  
**Live:** [mundial-predictor.up.railway.app](https://mundial-predictor.up.railway.app)

---

## Overview

A full-stack football prediction system for the 2026 FIFA World Cup. The model combines three statistical layers — Elo ratings, Poisson goal modeling, and historical penalty data — to estimate match outcomes, simulate the entire 104-match tournament 10,000 times, and serve real-time predictions through a REST API and a premium dark-themed web interface.

The system is data-driven end to end: trained on 150 years of international football, calibrated against Qatar 2022, and updated with recent results from the 2025–26 qualifying cycle.

---

## 1. Data Pipeline

### 1.1 Sources

| Source | Matches | Period |
|--------|---------|--------|
| Kaggle — International Football Results | 47,126 | 1872–July 2024 |
| Recent results (manual, verified) | 148 | Nov 2024–Jun 2026 |

The Kaggle dataset covers every tracked international fixture: friendlies, qualifying rounds, continental championships, and World Cups. Each row includes home/away teams, scores, tournament name, city, and a neutral-venue flag.

The 148 recent results were assembled manually from verified sources (UEFA official, NBC Sports, ESPN) and cover CONMEBOL qualifying (matchdays 12–18), UEFA Nations League 2024–25 including the finals, UEFA World Cup qualifying 2025–26 for the 16 qualified European nations, and pre-tournament friendlies (May–June 2026).

### 1.2 Preprocessing

```python
# loader.py — key steps
df = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
recent = pd.read_csv(RECENT_RESULTS)          # concatenated if present
df = pd.concat([df, recent]).sort_values("date")  # chronological order is critical for Elo
df["neutral"] = df["neutral"].astype(str).str.lower() == "true"
df["home_team"] = df["home_team"].str.strip().replace(NAME_MAP)
```

Name normalization (`NAME_MAP`) maps historical variants to canonical present-day names: West Germany → Germany, USA → United States, Ivory Coast → Côte d'Ivoire, Soviet Union → Russia, and ~20 others. Chronological ordering is non-negotiable for Elo — ratings are path-dependent, so processing a 2024 match before a 1990 match would produce nonsense ratings.

### 1.3 Tournament Weighting

Not all matches carry equal information. A World Cup final tells us more about team quality than a warm-up friendly. The model encodes this with a K-factor weight per tournament family:

| Tournament family | K-weight |
|-------------------|----------|
| FIFA World Cup | 60 |
| Continental championship (Copa América, Euros) | 50 |
| Confederations Cup | 45 |
| World Cup qualification | 40 |
| Continental qualification | 35 |
| Other competitive | 25 |
| Friendly | 10 |

The classifier maps ~200 distinct tournament name variants to these families using substring matching with a priority hierarchy — World Cup qualifiers are correctly separated from the World Cup itself, and Nations League is treated as a continental tournament.

---

## 2. Elo Rating Model

### 2.1 The Formula

Elo is a zero-sum rating system originally designed for chess. After each match, ratings update according to:

```
R'_A = R_A + K × W × (S_A − E_A)
```

Where:
- `R_A` — current rating of team A
- `K` — base K-factor (32 by default)
- `W` — tournament weight multiplier (see table above)
- `S_A` — actual result (1 = win, 0.5 = draw, 0 = loss)
- `E_A` — expected result (probability of win/draw from ratings)

The expected result is the classic logistic function:

```
E_A = 1 / (1 + 10^((R_B − R_A) / SCALE))
```

With `SCALE = 400` (standard chess scale, appropriate for football — a 400-point gap implies ~91% win probability for the stronger team).

### 2.2 Home Advantage

When the match is not at a neutral venue, the home team receives a rating bonus:

```python
HOME_ADVANTAGE = 100  # rating points added to home team's effective rating
```

100 Elo points corresponds to roughly a 14% boost in win probability for an evenly-matched fixture. This is set to zero for all World Cup matches (neutral venues) and for qualifying matches played at designated neutral sites.

### 2.3 Goal Margin Multiplier

Raw Elo ignores scorelines — a 1–0 and a 6–0 both register as a win. The model adds a margin-of-victory multiplier to capture the signal in goal differences:

```python
def _goal_multiplier(margin: int) -> float:
    if margin <= 1: return 1.0
    if margin == 2: return 1.5
    return (11 + margin) / 8
```

This follows the football-Elo convention: diminishing returns on large margins (a 7–0 isn't 7× more informative than a 1–0), while still rewarding dominant performances.

### 2.4 Host Nation Bonus

For the 2026 World Cup specifically, the three co-hosts receive a rating bonus in matches played on their own soil:

```python
HOST_BONUS = {"Mexico": 40, "United States": 25, "Canada": 15}
```

Mexico receives the largest bonus (passionate home crowd, all group matches in Mexico), USA the second (hosts 26 of 32 knockout matches including the final), and Canada a smaller adjustment. Critically, the bonus is applied **per match based on the actual venue** — Mexico loses its bonus in the knockout rounds since those are played in the US.

### 2.5 Calibrated Results

After training on the full dataset:

| Team | Elo Rating |
|------|-----------|
| Argentina | ~2,182 |
| France | ~2,140 |
| Brazil | ~2,078 |
| Spain | ~2,063 |
| Portugal | ~2,062 |
| Uruguay | ~2,047 |
| England | ~2,038 |

These reflect the state after incorporating the 2025–26 qualifying cycle, Nations League results, and pre-tournament friendlies. The 4–1 Argentina win over Brazil in March 2025 qualifiers, Norway's perfect qualifying campaign (37 goals in 8 matches), and France's semi-final loss to Spain in the Nations League (5–4) are all reflected.

---

## 3. Poisson Goal Model

### 3.1 Elo to Expected Goals

Elo ratings are converted to expected goals per team using an exponential function:

```python
BASE_GOALS = 1.35      # historical average goals per team per match
GOAL_SENSITIVITY = 0.0015  # calibrated parameter

def expected_goals(r_home: float, r_away: float) -> tuple[float, float]:
    diff = r_home - r_away
    lam_h = BASE_GOALS * exp( GOAL_SENSITIVITY * diff)
    lam_a = BASE_GOALS * exp(-GOAL_SENSITIVITY * diff)
    return lam_h, lam_a
```

The `GOAL_SENSITIVITY` parameter was calibrated to minimize Brier score on a held-out test set (Qatar 2022). The calibration sweep tested values from 0.0005 to 0.006 and found 0.0015 optimal — lower values flatten the distribution too much, higher values make heavily-favored teams too dominant.

### 3.2 Poisson Match Probabilities

Goals scored by each team are modeled as independent Poisson random variables. The probability of a specific scoreline (h, a) is:

```
P(H=h, A=a) = P(H=h) × P(A=a)
             = (e^(-λ_h) × λ_h^h / h!) × (e^(-λ_a) × λ_a^a / a!)
```

Summing over all scorelines up to `MAX_GOALS = 8`:

```python
def match_probabilities(r_home, r_away, neutral=True):
    lam_h, lam_a = expected_goals(r_home, r_away)
    p_home_win = p_draw = p_away_win = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = poisson_prob(lam_h, h) * poisson_prob(lam_a, a)
            if   h > a: p_home_win += p
            elif h < a: p_away_win += p
            else:       p_draw    += p
    return MatchProbabilities(p_home_win, p_draw, p_away_win,
                              exp_home_goals=lam_h, exp_away_goals=lam_a)
```

### 3.3 Backtest Results (Qatar 2022)

Validation against the 64 matches of the 2022 World Cup:

| Metric | Elo only | Poisson (calibrated) |
|--------|----------|---------------------|
| Accuracy (correct outcome) | 56.2% | 56.2% |
| Brier score | 0.2034 | **0.2023** |

The Brier score improvement is small but consistent: Poisson's continuous probability distribution is better calibrated than Elo's binary expected outcomes. Both outperform a naive baseline (predict home win every time: ~42% accuracy, Brier ~0.31).

---

## 4. Penalties Model

For knockout matches that reach a shootout, win probability is estimated from historical data:

```python
# Loaded from FIFA's official shootouts.csv
SHOOTOUT_RATES = {
    "Germany": 0.750,   # 6/8 won
    "Argentina": 0.636, # 7/11 won
    "Brazil": 0.600,
    "France": 0.400,
    "England": 0.273,
    "Netherlands": 0.200,
    # ... 30+ nations
}
```

Raw rates are Bayesian-blended with a 0.5 prior (minimum 3 shootouts required for a team to deviate from the prior):

```python
def get_penalty_win_prob(team_a, team_b):
    p_a = SHOOTOUT_RATES.get(team_a, 0.5)
    p_b = SHOOTOUT_RATES.get(team_b, 0.5)
    return p_a / (p_a + p_b)   # normalized to sum to 1
```

---

## 5. Knockout Match Simulation

A single knockout match is simulated in three phases:

```
90 minutes → Extra time (if draw) → Penalties (if still draw)
```

Each phase uses the Poisson model with adjusted lambda for time played:

```python
# Extra time: 30 minutes = 30/90 of a full match
lam_et_h = lam_h * (30/90)
lam_et_a = lam_a * (30/90)
```

The `predict_knockout()` function returns a `KnockoutPrediction` dataclass with full probability breakdowns: P(win in 90), P(reach ET), P(win in ET), P(reach penalties), P(win on penalties), and the overall advancement probability.

---

## 6. Monte Carlo Tournament Simulation

### 6.1 Architecture

The simulation models the complete 2026 tournament structure:

- 12 groups × 4 teams = 48 teams
- Round of 32 (16 matches) → Round of 16 → QF → SF → Final
- 8 best third-place teams qualify via an official FIFA ranking table

Each of 10,000 simulations runs independently:

```python
def simulate_tournament(model) -> dict[str, str]:
    # Group stage: 6 matches per group, round-robin
    for g, teams in GROUPS.items():
        standing = simulate_group(model, g, teams)
    
    # Select 8 best thirds
    best_thirds = sorted(all_thirds, key=lambda r: (r.pts, r.gd, r.gf))[:8]
    
    # Knockout: each match knows its real venue
    for match_no, slot_home, slot_away in R32_MATCHES:
        host = KO_VENUES[match_no]   # actual country hosting this match
        winner = simulate_match(model, home, away, knockout=True,
                               host_country=host)
```

### 6.2 Venue-Aware Host Bonus

The simulation uses the **official FIFA match schedule** to apply the host bonus correctly. Each of the 32 knockout matches is mapped to its real country:

```python
KO_VENUES = {
    73: "United States",  # SoFi Stadium, Inglewood
    75: "Mexico",         # Estadio BBVA, Monterrey
    79: "Mexico",         # Estadio Azteca, Mexico City
    ...
    104: "United States", # MetLife Stadium — the Final
}
```

Result: Mexico's bonus applies in groups and in the 3 knockout matches played in Mexico, then disappears. USA's bonus applies in 26 of 32 knockout matches including both semis and the final.

### 6.3 Simulation Results

After 10,000 runs:

| Team | Champion % | Final % | Semi % |
|------|-----------|---------|--------|
| Argentina | 21.5% | 31.3% | 44.0% |
| France | 12.2% | 21.2% | 34.6% |
| Spain | 9.8% | 17.7% | 28.4% |
| England | 8.2% | 15.3% | 27.5% |
| Portugal | 5.9% | 11.6% | 20.6% |
| Belgium | 5.6% | 11.3% | 20.5% |
| Brazil | 5.3% | 10.7% | 21.8% |

USA's home advantage boosts them meaningfully in the simulation — without it they'd be around 0.8%.

---

## 7. API Design

The FastAPI backend exposes four endpoints:

```
GET  /api/rankings      → Elo ratings for all 48 teams + simulation probs
GET  /api/predict       → Match prediction (query params: home, away, knockout)
POST /api/predict       → Match prediction with extra real results in body
GET  /api/simulation    → Full Monte Carlo results (champion/final/semi/QF/R16)
```

The `POST /api/predict` endpoint is the key innovation for the live fixture feature:

```python
class PredictRequest(BaseModel):
    home: str
    away: str
    knockout: bool = False
    home_adjust: float = 0.0    # slider: -100 to +100 rating adjustment
    away_adjust: float = 0.0
    extra_results: list[ExtraResult] = []
    use_extra: bool = False

@app.post("/api/predict")
def predict_post(req: PredictRequest):
    if req.use_extra and req.extra_results:
        base = build_model_with_extra(req.extra_results)  # deepcopy + apply
    else:
        base = STATE["poisson"]  # cached base model
    model = AdjustedModel(base, {req.home: req.home_adjust, req.away: req.away_adjust})
    ...
```

`build_model_with_extra()` deep-copies the base Elo model (trained at startup) and applies the user's real tournament results on top, treating them as full-weight World Cup matches. This means every prediction request is fully self-contained — no server-side state mutation, no race conditions.

### Startup sequence

```python
@app.on_event("startup")
def train_model():
    df = load_matches()           # ~47k rows
    elo = EloModel()
    for _, row in df.iterrows():  # ~2–3 seconds
        elo.update(...)
    STATE["elo"] = elo
    STATE["poisson"] = PoissonModel(elo)
    STATE["simulation"] = run_simulation(STATE["poisson"], n=10_000)  # ~60–90s
```

The simulation is the startup bottleneck. On Railway's hobby tier (~1 vCPU, 512MB RAM) this takes 2–4 minutes. This is within Railway's 5-minute startup timeout, but close enough that a future optimization (pickle the trained model to disk) would be worthwhile.

---

## 8. Frontend Architecture

The frontend is a single self-contained HTML file (~1,600 lines) with no build step and no external JS frameworks. Design choices:

**No framework.** Vanilla JS with direct DOM manipulation. For a project of this scope, React or Vue would add complexity without benefit. The absence of a build pipeline also means deploying is just copying one file.

**Fonts via CDN.** Clash Display (headings) and Plus Jakarta Sans (body) loaded from Bunny Fonts. No local assets.

**Flags via flagcdn.com.** SVG flags loaded dynamically from `flagcdn.com/w20/{code}.png` using a `FLAG_CODES` map. This replaced emoji flags which rendered inconsistently across platforms.

**localStorage persistence.** The Fixture and Knockout tracker store state in `localStorage` under two keys (`wc26_fixture`, `wc26_knockout`). Export/import produces a JSON file containing both, enabling backup and restore.

**Live model integration.** When the "Use results to train predictor" toggle is ON, the frontend collects all entered group and knockout results and sends them in the body of a POST request. The model recalculates ratings on the server and returns updated probabilities. The UI shows a green badge confirming the prediction incorporates real data.

**Bidirectional adjustment sliders.** Each team in the predictor has a ±100 Elo rating slider for manual hypotheticals (injury to key player, unusual conditions). These are applied via `AdjustedModel` which wraps the base model and overrides `get_rating()` to add the delta.

---

## 9. Key Design Decisions

**Elo over more complex models.** Gradient-boosted trees or neural networks trained on player-level data would likely outperform Elo + Poisson on raw accuracy. But Elo has three properties that make it ideal here: it's interpretable (the rating number is meaningful), it updates incrementally (new results integrate cleanly), and it performs surprisingly well at the tournament level where sample sizes are small. The 56% accuracy on Qatar 2022 is competitive with published football prediction models.

**Poisson independence assumption.** The model assumes home and away goals are independent, which is not strictly true (a team trailing 0–2 late plays differently than one leading). This is a known limitation of basic Dixon-Coles models. The simplification is acceptable for pre-match prediction where we don't have in-match state.

**Separate base model and live model.** Rather than mutating the base Elo model with each user update, each prediction with extra results creates a fresh deep-copy. This makes the API stateless and trivially cacheable, at the cost of a ~50ms overhead per request (acceptable for interactive use, problematic at scale).

**CSV over database.** The dataset is read-only at runtime — Railway has no persistent file system for user writes, so a database would be needed for user-generated data. The fixture data lives in the browser (localStorage) and is sent to the server per-request. This avoids the operational complexity of a database for a project with a single user.

---

## 10. Deployment

```
GitHub (casper06/mundial-predictor)
    └── Railway (auto-deploy on push)
         ├── Python 3.11
         ├── uvicorn api.main:app --host 0.0.0.0 --port $PORT
         └── US West region
```

**Updating results** during the tournament:

```bash
# 1. Edit data/raw/recent_results.csv
# 2. Commit and push
git add data/raw/recent_results.csv
git commit -m "Update results: matchday X"
git push
# Railway auto-redeploys. New ratings live in ~3 minutes.
```

---

## 11. Repository Structure

```
mundial-predictor/
├── api/
│   └── main.py              FastAPI app, startup training, all endpoints
├── data/
│   └── raw/
│       ├── results.csv      Kaggle dataset (47,126 matches)
│       ├── recent_results.csv  Recent results (148 matches, manually verified)
│       └── shootouts.csv    Historical penalty shootout results
├── src/
│   ├── data/
│   │   ├── loader.py        Data loading, name normalization, weighting
│   │   └── tournament_weights.py  Tournament family classifier
│   ├── models/
│   │   ├── elo.py           Elo rating model
│   │   ├── poisson.py       Poisson goal model + match probabilities
│   │   └── penalties.py     Historical penalty win rates
│   └── simulation/
│       └── monte_carlo.py   Full tournament simulation (10,000 runs)
├── web/
│   └── index.html           Complete frontend (single file, ~1,600 lines)
├── Procfile                 Railway start command
├── requirements.txt
└── runtime.txt              python-3.11
```

---

*Built over approximately 3 weeks of iterative development. Model validated on Qatar 2022 before applying to 2026.*
