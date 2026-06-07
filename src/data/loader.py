"""
Loads and normalises the Kaggle international football results dataset.
"""
from pathlib import Path
import pandas as pd
from .tournament_weights import get_weight, classify_tournament

ROOT         = Path(__file__).resolve().parents[2]
RAW_RESULTS  = ROOT / "data" / "raw" / "results.csv"
PROCESSED    = ROOT / "data" / "processed" / "matches.parquet"

NAME_MAP = {
    "USA":                   "United States",
    "West Germany":          "Germany",
    "Soviet Union":          "Russia",
    "Czechoslovakia":        "Czech Republic",
    "Republic of Ireland":   "Ireland",
    "Ivory Coast":           "Côte d'Ivoire",
    "South Korea":           "Korea Republic",
    "Macedonia":             "North Macedonia",
    "FYR Macedonia":         "North Macedonia",
    "Swaziland":             "Eswatini",
    "Cape Verde Islands":    "Cape Verde",
    "Zaire":                 "DR Congo",
    "Congo DR":              "DR Congo",
    "Congo":                 "Republic of Congo",
    "Trinidad and Tobago":   "Trinidad & Tobago",
    "East Timor":            "Timor-Leste",
    "Taiwan":                "Chinese Taipei",
}

def load_matches(path: Path = RAW_RESULTS) -> pd.DataFrame:
    """
    Load, clean and enrich the raw results CSV.
    Also concatenates data/raw/recent_results.csv if it exists (friendlies,
    World Cup matches entered manually before/during the tournament).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}\n"
            "Place results.csv in data/raw/"
        )

    df = pd.read_csv(path, parse_dates=["date"])

    # Concatenate recent results if the file exists
    recent_path = path.parent / "recent_results.csv"
    if recent_path.exists():
        recent = pd.read_csv(recent_path, parse_dates=["date"])
        if len(recent):
            df = pd.concat([df, recent], ignore_index=True)

    # basic cleaning
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"]    = df["neutral"].astype(str).str.strip().str.lower() == "true"

    # name normalisation
    df["home_team"] = df["home_team"].str.strip().replace(NAME_MAP)
    df["away_team"] = df["away_team"].str.strip().replace(NAME_MAP)

    # tournament enrichment
    df["tournament_family"] = df["tournament"].apply(classify_tournament)
    df["k_weight"]          = df["tournament"].apply(get_weight)

    # sort chronologically (Elo must process in time order)
    df = df.sort_values("date").reset_index(drop=True)

    return df

def save_processed(df: pd.DataFrame, path: Path = PROCESSED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} matches to {path}")

if __name__ == "__main__":
    df = load_matches()
    print(f"{len(df):,} matches | {df['date'].min().date()} → {df['date'].max().date()}")
    print(df["tournament_family"].value_counts().to_string())
    save_processed(df)