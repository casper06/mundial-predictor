"""
Asigna a cada partido un peso K segun la importancia del torneo.

El peso K controla cuanto mueve un partido el rating Elo:
  - K alto  -> el resultado importa mucho (mundiales, finales)
  - K bajo  -> el resultado importa poco (amistosos)

Esto es lo que nos permite "aprender a discriminar" amistosos de torneos:
el sistema confia mucho mas en lo que pasa en una Copa del Mundo que en un
amistoso donde se prueban suplentes.

El campo 'tournament' viene del dataset de Kaggle. Hay cientos de valores
distintos, asi que los agrupamos por familia con coincidencia de texto.
"""

# Pesos base por familia de torneo. Ordenado de mas a menos importante.
# Estos numeros son un punto de partida razonable; los vamos a calibrar
# despues con el backtest contra mundiales pasados.
WEIGHT_TABLE = {
    "world_cup": 60,        # Copa del Mundo (fase final)
    "world_cup_qualif": 40, # Eliminatorias mundialistas
    "continental": 50,      # Copa America, Euro, etc. (fase final)
    "continental_qualif": 35,
    "confederations": 45,   # Copa Confederaciones, Nations League
    "friendly": 10,         # Amistosos: peso minimo
    "other": 25,            # Torneos menores no clasificados
}


def classify_tournament(name: str) -> str:
    """Devuelve la familia de torneo a partir del nombre crudo del dataset."""
    n = (name or "").strip().lower()

    if n == "friendly":
        return "friendly"

    # Mundial
    if "fifa world cup" in n:
        if "qualif" in n:
            return "world_cup_qualif"
        return "world_cup"

    # Torneos continentales (fase final)
    continental_keys = [
        "uefa euro", "copa américa", "copa america", "african cup",
        "afc asian cup", "gold cup", "concacaf championship",
        "oceania nations cup", "ofc nations cup",
    ]
    if any(k in n for k in continental_keys):
        if "qualif" in n:
            return "continental_qualif"
        return "continental"

    if "qualif" in n:
        return "continental_qualif"

    # Confederaciones / Nations League
    if "confederations" in n or "nations league" in n:
        return "confederations"

    return "other"


def get_weight(tournament_name: str) -> int:
    """Peso K final para un partido dado el nombre de su torneo."""
    family = classify_tournament(tournament_name)
    return WEIGHT_TABLE[family]
