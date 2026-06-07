"""
Interfaz comun para todos los modelos de prediccion.

Hoy implementamos Elo. Manana podemos implementar un modelo de ML
(gradient boosting, etc.). Mientras respeten esta interfaz, la capa de
simulacion y la API no se enteran del cambio: solo piden predict_match().
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MatchPrediction:
    """Resultado de predecir un partido."""
    p_home_win: float   # probabilidad de que gane el local
    p_draw: float       # probabilidad de empate
    p_away_win: float   # probabilidad de que gane el visitante
    exp_home_goals: float = 0.0   # goles esperados del local (lo llena Poisson)
    exp_away_goals: float = 0.0   # goles esperados del visitante


class BaseModel(ABC):
    @abstractmethod
    def predict_match(self, home: str, away: str, neutral: bool = False) -> MatchPrediction:
        """Predice un partido entre dos selecciones."""
        ...

    @abstractmethod
    def get_rating(self, team: str) -> float:
        """Devuelve la fuerza actual de una seleccion."""
        ...
