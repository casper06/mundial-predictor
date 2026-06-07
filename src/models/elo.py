"""
Motor Elo adaptado al futbol de selecciones.

Sobre el Elo clasico de ajedrez agrega tres ajustes propios del futbol:
  1. Margen de gol  -> las goleadas mueven mas el rating
  2. Localia        -> jugar de local da ventaja (cero en cancha neutral)
  3. Peso de torneo -> un amistoso mueve poco; una final de mundial, mucho

El rating no necesita saber nada de planteles ni lesiones para funcionar.
Esos datos se suman despues como un ajuste (delta) sobre el rating base.
"""
from collections import defaultdict

from .base import BaseModel, MatchPrediction
from ..data.tournament_weights import get_weight

# Rating inicial de una seleccion de la que no sabemos nada todavia.
DEFAULT_RATING = 1500.0

# Cuanta ventaja (en puntos de rating) da jugar de local. Cero si es neutral.
HOME_ADVANTAGE = 100.0

# Constante de escala del Elo. 400 es el estandar historico.
SCALE = 400.0


class EloModel(BaseModel):
    def __init__(self, default_rating: float = DEFAULT_RATING,
                 home_advantage: float = HOME_ADVANTAGE):
        self.ratings: dict[str, float] = defaultdict(lambda: default_rating)
        self.home_advantage = home_advantage
        self._default = default_rating

    # ---- consulta ----
    def get_rating(self, team: str) -> float:
        return self.ratings[team]

    def _expected(self, r_a: float, r_b: float) -> float:
        """Probabilidad esperada de que A le gane a B (sin contar empates)."""
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / SCALE))

    # ---- entrenamiento: procesar un partido jugado ----
    def update(self, home: str, away: str, home_goals: int, away_goals: int,
               tournament: str = "friendly", neutral: bool = False) -> None:
        """Actualiza los ratings con el resultado de un partido real."""
        r_home = self.ratings[home]
        r_away = self.ratings[away]

        # Localia: solo afecta la expectativa, no el rating guardado.
        bonus = 0.0 if neutral else self.home_advantage
        exp_home = self._expected(r_home + bonus, r_away)

        # Resultado real desde la perspectiva del local: 1 gana, 0.5 empata, 0 pierde.
        if home_goals > away_goals:
            score_home = 1.0
        elif home_goals == away_goals:
            score_home = 0.5
        else:
            score_home = 0.0

        # K base segun importancia del torneo.
        k = get_weight(tournament)

        # Ajuste por margen de gol (formula del ranking World Football Elo).
        k *= self._goal_multiplier(abs(home_goals - away_goals))

        # El intercambio es de suma cero: lo que gana uno lo pierde el otro.
        delta = k * (score_home - exp_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta

    @staticmethod
    def _goal_multiplier(margin: int) -> float:
        """Multiplicador del K segun la diferencia de goles."""
        if margin <= 1:
            return 1.0
        if margin == 2:
            return 1.5
        # 3 o mas: crece pero cada vez menos (rendimientos decrecientes).
        return (11 + margin) / 8.0

    # ---- prediccion ----
    def predict_match(self, home: str, away: str, neutral: bool = False) -> MatchPrediction:
        r_home = self.ratings[home]
        r_away = self.ratings[away]
        bonus = 0.0 if neutral else self.home_advantage

        # Probabilidad de victoria del local sin empates.
        p_home_raw = self._expected(r_home + bonus, r_away)

        # El Elo puro no modela el empate. Lo estimamos: cuanto mas parejo
        # el partido, mas probable el empate. Esto es una aproximacion;
        # Poisson (proximo paso) lo va a hacer mucho mejor.
        diff = abs((r_home + bonus) - r_away)
        p_draw = 0.28 * (1.0 - min(diff / 600.0, 0.9))

        p_home = p_home_raw * (1.0 - p_draw)
        p_away = (1.0 - p_home_raw) * (1.0 - p_draw)

        return MatchPrediction(
            p_home_win=round(p_home, 4),
            p_draw=round(p_draw, 4),
            p_away_win=round(p_away, 4),
        )
