import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

from src.data.loader import load_matches
from src.models.elo import EloModel
from src.models.poisson import PoissonModel

df = load_matches()
elo = EloModel()
for _, row in df.iterrows():
    elo.update(row['home_team'], row['away_team'],
               row['home_score'], row['away_score'],
               row['tournament'], row['neutral'])

model = PoissonModel(elo)

pairs = [
    ('Mexico',    'South Africa'),
    ('Argentina', 'Algeria'),
    ('Brazil',    'Morocco'),
    ('France',    'Senegal'),
    ('England',   'Croatia'),
    ('Spain',     'Cape Verde'),
]

print(f"{'Match':<30} {'Home win':>9} {'Draw':>7} {'Away win':>9}  xG")
print('-' * 70)
for home, away in pairs:
    p    = model.predict_match(home, away, neutral=True)
    tops = model.top_scorelines(home, away, neutral=True, top_n=3)
    scores = ', '.join(f"{s['score']}({s['probability']:.0%})" for s in tops)
    print(f"{home} vs {away:<18} {p.p_home_win:>8.1%} {p.p_draw:>6.1%} {p.p_away_win:>8.1%}  {p.exp_home_goals:.2f}-{p.exp_away_goals:.2f}")
    print(f"  -> {scores}")