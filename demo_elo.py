"""
Demo del motor Elo con partidos reales (cargados a mano).

Cuando tengamos el CSV de Kaggle, en vez de esta lista vamos a recorrer
las ~48.000 filas del dataset. La logica de entrenamiento es identica.

Correr con:  python demo_elo.py
"""
import sys
from pathlib import Path

# Permite importar el paquete src/ sin instalar nada.
sys.path.insert(0, str(Path(__file__).parent))

from src.models.elo import EloModel

# (local, visitante, goles_local, goles_visitante, torneo, neutral)
# Una muestra chica de partidos reales para ver el motor en accion.
PARTIDOS = [
    ("Argentina", "France", 4, 2, "FIFA World Cup", True),    # final 2022 (definicion en penales contada como victoria)
    ("Argentina", "Croatia", 3, 0, "FIFA World Cup", True),   # semi 2022
    ("France",    "Morocco", 2, 0, "FIFA World Cup", True),   # semi 2022
    ("Argentina", "Netherlands", 2, 2, "FIFA World Cup", True),
    ("Brazil",    "Argentina", 1, 0, "FIFA World Cup qualification", False),
    ("Argentina", "Brazil", 1, 0, "Copa América", True),
    ("Germany",   "Argentina", 1, 0, "FIFA World Cup", True),
    ("Spain",     "Germany", 2, 1, "Friendly", False),
    ("Argentina", "Panama", 5, 0, "Friendly", False),
]


def main():
    model = EloModel()

    print("Entrenando con", len(PARTIDOS), "partidos...\n")
    for home, away, hg, ag, tour, neutral in PARTIDOS:
        model.update(home, away, hg, ag, tournament=tour, neutral=neutral)

    print("Ratings resultantes (de mayor a menor):")
    ranking = sorted(model.ratings.items(), key=lambda kv: kv[1], reverse=True)
    for team, rating in ranking:
        print(f"  {team:14s} {rating:7.1f}")

    print("\nPrediccion Argentina vs France (cancha neutral, tipo final):")
    pred = model.predict_match("Argentina", "France", neutral=True)
    print(f"  Gana Argentina: {pred.p_home_win:5.1%}")
    print(f"  Empate:         {pred.p_draw:5.1%}")
    print(f"  Gana France:    {pred.p_away_win:5.1%}")


if __name__ == "__main__":
    main()
