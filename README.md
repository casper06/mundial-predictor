# Mundial Predictor

Predictor del Mundial 2026 basado en 150 años de partidos internacionales.

## Estado
- [x] Pipeline de datos + normalización de nombres
- [x] Motor Elo (margen de gol, localía, peso de torneo)
- [x] Poisson calibrado (marcadores + probabilidades)
- [x] Modelo de penales (datos históricos de shootouts)
- [x] Flujo knockout: 90min → tiempo extra → penales
- [x] Backtest contra Qatar 2022 (56.2% accuracy, Brier 0.2023)
- [x] Backend FastAPI (sirve API + web)
- [x] Web con predictor conectado al backend
- [ ] Variables de usuario (lesiones, forma) — PRÓXIMO
- [ ] Simulación Monte Carlo en la web
- [ ] Scraping de planteles

## Cómo correr

### 1. Instalar dependencias (una vez)
    pip install -r requirements.txt

### 2. Datos
Poné results.csv y shootouts.csv en data/raw/ (del dataset de Kaggle)

### 3. Levantar el servidor web
    uvicorn api.main:app --reload

Después abrí http://localhost:8000

### Scripts sueltos (sin servidor)
    python train_elo.py                    # ver ranking
    python -m src.evaluation.backtest      # validar modelo
    python predict_knockout.py             # probar knockouts

## Arquitectura
- data/raw/       resultados de partidos (Kaggle)
- data/context/   planteles, lesiones (futuro)
- src/models/     elo, poisson, penalties, knockout
- api/main.py     FastAPI: entrena al arrancar, expone /api/* y sirve la web
- web/index.html  frontend (usa la API, con fallback a datos embebidos)

El frontend pide datos a la API en vivo. Si lo abrís como archivo suelto
(doble clic) sin servidor, usa datos embebidos como fallback.
