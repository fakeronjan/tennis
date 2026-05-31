"""
tennis_engine.py — fakeronjan WLS rating engine for tennis (singles, ATP + WTA, 1973+).

Pipeline:
  1. download_sackmann()   — fetch & cache Jeff Sackmann's tennis_atp + tennis_wta CSVs
  2. build_unified()       — parse + normalize into all_matches.csv (one row per match)
  3. build_observations()  — one row per SET, response = signed sqrt(game margin)
  4. solve_ratings()       — WLS solve: base rating + 4 surface deltas, per tour separately
  5. write_outputs()       — ratings CSVs + JSON snapshots for the SPA

Data source: Jeff Sackmann (CC-BY-NC-SA 4.0). Attribution required.
  https://github.com/JeffSackmann/tennis_atp
  https://github.com/JeffSackmann/tennis_wta
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pandas as pd


# ============================================================
# PARAMETERS
# ============================================================

MIN_YEAR = 1973   # Open Era data is patchy 1968-72; Sackmann coverage solidifies in '73
DATA_DIR = Path(__file__).parent / "data" / "sackmann"
ALL_MATCHES_CSV = Path(__file__).parent / "all_matches.csv"

# Tournament tier weights — mirrors MESSI's continental boost / ZIDANE's UCL boost.
# Sackmann's tourney_level codes:
#   G  = Grand Slam
#   M  = ATP Masters 1000 / WTA Premier Mandatory + Premier 5 (lumping the "1000s")
#   P  = WTA Premier (1000-tier, older code)
#   PM = WTA Premier Mandatory
#   A  = ATP 500/250 / WTA International (mid-tier)
#   I  = WTA International (250-tier)
#   F  = ATP Finals / WTA Finals (year-end championships)
#   D  = Davis Cup / Fed Cup / Billie Jean King Cup (team events)
#   C  = Challenger
#   S  = ATP Satellite
#   T1/T2/T3/T4/T5 = WTA older tier codes (T1=1000-ish, descending)
TIER_WEIGHTS = {
    "G":  2.0,   # Grand Slam
    "F":  1.5,   # Year-end Finals
    "M":  1.5,   # ATP/WTA 1000s
    "P":  1.5,   # WTA older Premier (1000-ish)
    "PM": 1.5,   # WTA Premier Mandatory
    "T1": 1.5,   # WTA T1 (1000-tier)
    "A":  1.0,   # ATP 500/250 (we don't separate further — Sackmann doesn't always)
    "I":  0.8,   # WTA International (250-tier)
    "T2": 1.0,   # WTA T2
    "T3": 0.8,   # WTA T3
    "T4": 0.8,   # WTA T4
    "T5": 0.8,   # WTA T5
    "D":  1.0,   # Davis Cup / Fed Cup (per user 2026-05-30)
    # Excluded: Challenger (C), Satellite (S), Futures
}

# Surfaces. Indoor and outdoor hard are combined per industry convention.
# Carpet retired around 2009 but had a real role in '70s-'90s indoor events.
SURFACES = {"Hard", "Clay", "Grass", "Carpet"}


# ============================================================
# STEP 1 — DOWNLOAD SACKMANN
# ============================================================

USER_AGENT = "Mozilla/5.0 (compatible; fakeronjan tennis ratings bot)"


def _http_get(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_sackmann(min_year: int = MIN_YEAR, refresh_latest: int = 2) -> None:
    """Download (and cache) Jeff Sackmann's per-year ATP + WTA singles match CSVs.

    Idempotent: skips files already on disk except for the latest `refresh_latest`
    years (always re-fetched to absorb new matches mid-season).

    Layout written:
      data/sackmann/atp_matches_<YEAR>.csv
      data/sackmann/wta_matches_<YEAR>.csv
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    this_year = datetime.now().year
    refresh_from = this_year - refresh_latest + 1

    for tour, base in [
        ("atp", "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"),
        ("wta", "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"),
    ]:
        for year in range(min_year, this_year + 1):
            fname = f"{tour}_matches_{year}.csv"
            path = DATA_DIR / fname
            if path.exists() and year < refresh_from:
                continue
            url = f"{base}/{fname}"
            try:
                print(f"  fetching {tour.upper()} {year}...", end=" ", flush=True)
                data = _http_get(url)
                path.write_bytes(data)
                print(f"{len(data):,} bytes")
            except HTTPError as e:
                if e.code == 404:
                    print("(not yet posted)")
                else:
                    print(f"HTTP {e.code}")
            except Exception as e:
                print(f"FAILED ({e})")


# ============================================================
# STEP 2 — BUILD UNIFIED all_matches.csv
# ============================================================

def _parse_date(raw) -> pd.Timestamp:
    """Sackmann's tourney_date is an int YYYYMMDD (e.g. 20210712)."""
    s = str(int(raw)) if pd.notna(raw) else ""
    if len(s) == 8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.NaT


def _load_one(path: Path, tour: str) -> pd.DataFrame:
    """Load one year-CSV, return normalized DataFrame (or empty if nothing usable)."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception as e:
        print(f"  ! could not read {path.name}: {e}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df["tour"] = tour.upper()  # "ATP" or "WTA"
    df["date"] = df["tourney_date"].apply(_parse_date)
    # Drop rows with no parseable date or no winner/loser name
    df = df.dropna(subset=["date"])
    df = df[df["winner_name"].notna() & df["loser_name"].notna()].copy()
    return df


def build_unified(out_path: Path = ALL_MATCHES_CSV) -> pd.DataFrame:
    """Load all cached Sackmann CSVs, concat into one DataFrame, write all_matches.csv.

    Filters:
      - Tier in TIER_WEIGHTS (excludes Challenger/Satellite/Futures)
      - Surface in SURFACES (normalizes 'I.Hard' style variants if any)
      - Walkovers (W/O) dropped
      - Drops matches with no score string at all
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found — run download_sackmann() first."
        )

    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        tour = "ATP" if path.name.startswith("atp_") else "WTA"
        d = _load_one(path, tour)
        if not d.empty:
            frames.append(d)

    if not frames:
        raise RuntimeError("No Sackmann data loaded. Did download_sackmann run?")

    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    print(f"\nLoaded {len(df):,} raw matches from {len(frames)} files.")

    # Normalize surface (Sackmann is usually consistent but defensive parse)
    df["surface"] = df["surface"].fillna("").str.strip()
    # Filter to known surfaces
    df = df[df["surface"].isin(SURFACES)].copy()

    # Tier filter + weight
    df["tier"] = df["tourney_level"].fillna("").str.strip()
    df = df[df["tier"].isin(TIER_WEIGHTS.keys())].copy()
    df["tier_weight"] = df["tier"].map(TIER_WEIGHTS)

    # Drop walkovers and empty-score rows
    df["score"] = df["score"].fillna("").str.strip()
    df = df[df["score"] != ""].copy()
    df = df[~df["score"].str.contains("W/O", case=False, na=False)].copy()

    # best_of as int (default 3 if missing)
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)

    # Keep only the columns we actually need for the engine
    keep = [
        "date", "tour", "tier", "tier_weight", "surface", "best_of", "round",
        "tourney_name", "tourney_id",
        "winner_id", "winner_name", "loser_id", "loser_name",
        "score",
    ]
    df = df[keep].sort_values("date").reset_index(drop=True)

    # Stats summary before writing
    print(f"After filters: {len(df):,} matches "
          f"({(df['tour'] == 'ATP').sum():,} ATP, {(df['tour'] == 'WTA').sum():,} WTA)")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Surfaces:\n{df['surface'].value_counts().to_string()}")
    print(f"Tiers:\n{df['tier'].value_counts().to_string()}")

    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(df):,} matches")
    return df


# ============================================================
# STEP 3 — PARSE SCORES + BUILD SET-LEVEL OBSERVATIONS
# ============================================================

import re

# Captures the two integer game counts from a set token like "6-4" or "7-6(3)".
_SET_TOKEN_RE = re.compile(r"^(\d+)-(\d+)")
# Tokens that mark the end of meaningful play but aren't sets themselves.
_NON_SET_TOKENS = {"RET", "DEF", "W/O", "WALK", "ABN", "ABD", "Played"}


def parse_score(score_str: str) -> list[tuple[int, int]]:
    """Parse a Sackmann score string into a list of (winner_games, loser_games) tuples.

    Examples:
      "6-4 6-2"               → [(6,4), (6,2)]
      "6-4 6-7(3) 6-3"        → [(6,4), (6,7), (6,3)]
      "6-7(1) 7-5 6-4 1-0 RET" → [(6,7), (7,5), (6,4), (1,0)]    # retirement: keep partial set
      ""                       → []
    """
    if not score_str or pd.isna(score_str):
        return []
    out: list[tuple[int, int]] = []
    for token in str(score_str).split():
        if token.upper() in _NON_SET_TOKENS:
            continue
        m = _SET_TOKEN_RE.match(token)
        if not m:
            continue
        w, l = int(m.group(1)), int(m.group(2))
        # Sanity guard: any set with both sides at 0 is unparseable
        if w == 0 and l == 0:
            continue
        out.append((w, l))
    return out


def build_observations(matches: pd.DataFrame) -> pd.DataFrame:
    """Expand match-level DataFrame to one row per set.

    Each set becomes a regression observation:
      - X: row of zeros except +1 for the set winner's base + winner's surface, -1 for loser
      - y: sqrt(games_diff)  (signed implicitly by which side's +1 is in X)
      - w: tier_weight × recency_weight (recency applied per-snapshot, not here)
    """
    rows = []
    for r in matches.itertuples(index=False):
        sets = parse_score(r.score)
        if not sets:
            continue
        for set_num, (winner_games, loser_games) in enumerate(sets, start=1):
            margin = winner_games - loser_games
            # Skip nonsense (e.g. both 0 — already filtered, defensive)
            if margin <= 0:
                # Score string has set written from match-winner POV first,
                # so winner_games >= loser_games. If reversed (rare bad parse),
                # treat as the set loser winning that set.
                # Keep the row but flip winner/loser for THIS set.
                rows.append({
                    "date": r.date,
                    "tour": r.tour,
                    "surface": r.surface,
                    "tier_weight": r.tier_weight,
                    "winner": r.loser_name,
                    "loser": r.winner_name,
                    "winner_id": r.loser_id,
                    "loser_id": r.winner_id,
                    "set_num": set_num,
                    "games_diff": -margin,
                    "y": np.sqrt(-margin),
                })
                continue
            rows.append({
                "date": r.date,
                "tour": r.tour,
                "surface": r.surface,
                "tier_weight": r.tier_weight,
                "winner": r.winner_name,
                "loser": r.loser_name,
                "winner_id": r.winner_id,
                "loser_id": r.loser_id,
                "set_num": set_num,
                "games_diff": margin,
                "y": np.sqrt(margin),
            })
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out


# ============================================================
# STEP 4 — WLS SOLVE (PER TOUR, SINGLE SNAPSHOT)
# ============================================================
# Each set is one observation. Each player gets 4 unknowns: a base rating + 3
# surface deltas (sum to zero per player as a soft constraint via a tiny ridge,
# plus per-tour zero-sum constraint on the base rating).
#
# We solve men's and women's tours independently — they don't co-mingle.

SURFACE_LIST = ["Hard", "Clay", "Grass", "Carpet"]


MIN_SETS_PLAYED = 30  # filter for published rankings — drops tiny-sample noise

def solve_tour(obs: pd.DataFrame, snapshot_date: pd.Timestamp,
               window_days: int = 365,
               min_sets_played: int = MIN_SETS_PLAYED) -> pd.DataFrame:
    """Run a single WLS solve on the rolling window ending at snapshot_date.

    Returns a DataFrame: one row per player who meets min_sets_played, with columns
      [player, base, hard_delta, clay_delta, grass_delta, carpet_delta, sets_played]

    Implementation:
      - Filter obs to window [snapshot_date - window_days, snapshot_date]
      - Apply linear recency decay: weight ramps from ~0 at window edge to 1 at snapshot
      - Build X with 5 columns per player: base + 4 surface indicators
      - Solve via numpy.linalg.lstsq on weighted system
      - Enforce zero-sum on base ratings via a high-weight extra row
      - Filter output to players with at least min_sets_played sets in window
    """
    snapshot_date = pd.Timestamp(snapshot_date)
    window_start = snapshot_date - pd.Timedelta(days=window_days)
    w = obs[(obs["date"] >= window_start) & (obs["date"] <= snapshot_date)].copy()
    if w.empty:
        return pd.DataFrame()

    # Linear recency decay: most-recent set weight = 1, oldest in window = ~0
    days_back = (snapshot_date - w["date"]).dt.days.to_numpy(dtype=float)
    recency = np.maximum(0.0, 1.0 - days_back / window_days)
    sample_weight = w["tier_weight"].to_numpy(dtype=float) * recency

    # Build player index
    players = sorted(set(w["winner"]) | set(w["loser"]))
    pidx = {p: i for i, p in enumerate(players)}
    n_p = len(players)
    n_obs = len(w)
    if n_p < 4 or n_obs < 50:
        return pd.DataFrame()

    # Column layout: 5 columns per player
    #   [base, hard, clay, grass, carpet]
    # Order matches SURFACE_LIST for the deltas.
    SURF_COL = {s: i + 1 for i, s in enumerate(SURFACE_LIST)}  # +1 because base=0
    n_cols = n_p * 5

    # Build X sparsely-ish (still dense numpy — for our 1-year window scale it fits)
    # Each row: +1 on winner.base, -1 on loser.base, +1 on winner.<surface>, -1 on loser.<surface>
    X = np.zeros((n_obs + 1, n_cols), dtype=np.float32)
    y = np.zeros(n_obs + 1, dtype=np.float32)
    sw = np.zeros(n_obs + 1, dtype=np.float32)

    winners = w["winner"].to_numpy()
    losers = w["loser"].to_numpy()
    surfaces = w["surface"].to_numpy()
    y_obs = w["y"].to_numpy(dtype=np.float32)

    for i in range(n_obs):
        wi = pidx[winners[i]]
        li = pidx[losers[i]]
        # Base rating diff
        X[i, wi * 5] = 1.0
        X[i, li * 5] = -1.0
        # Surface delta diff
        s_col = SURF_COL[surfaces[i]]
        X[i, wi * 5 + s_col] = 1.0
        X[i, li * 5 + s_col] = -1.0
        y[i] = y_obs[i]
        sw[i] = sample_weight[i]

    # Zero-sum constraint on BASE ratings only (very high weight)
    # This anchors the regression — sum of base ratings = 0.
    for p in range(n_p):
        X[n_obs, p * 5] = 1.0
    y[n_obs] = 0.0
    sw[n_obs] = 1e7

    # Apply weights and solve
    sqrt_w = np.sqrt(sw)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    # Reshape into (n_p, 5)
    R = r.reshape(n_p, 5)

    # Games played per player (in window)
    pg = pd.concat([
        w[["winner"]].rename(columns={"winner": "player"}),
        w[["loser"]].rename(columns={"loser": "player"}),
    ])
    games_played = pg["player"].value_counts().to_dict()

    out = pd.DataFrame({
        "player":       players,
        "base":         R[:, 0],
        "hard_delta":   R[:, 1],
        "clay_delta":   R[:, 2],
        "grass_delta":  R[:, 3],
        "carpet_delta": R[:, 4],
        "sets_played":  [games_played.get(p, 0) for p in players],
    })
    # Filter to players with enough data in window
    out = out[out["sets_played"] >= min_sets_played].copy()
    return out.sort_values("base", ascending=False).reset_index(drop=True)


# ============================================================
# STEP 5 — ROLLING SNAPSHOTS + JSON OUTPUTS
# ============================================================
# Snapshot cadence: end-of-November each year (after ATP/WTA Finals usually) +
# current state. ~52 snapshots × 2 tours = ~100 solves, ~5-10 minutes total.

import json

DOCS_DATA = Path(__file__).parent / "docs" / "data"
RATINGS_CSV = Path(__file__).parent / "tennis_ratings.csv.gz"


def build_rolling_snapshots(obs: pd.DataFrame) -> pd.DataFrame:
    """Solve ratings at end-of-November each year + the latest date in data.

    Returns long-format DataFrame: one row per (snapshot_date, tour, player).
    """
    min_date = obs["date"].min()
    max_date = obs["date"].max()
    snap_dates = []
    for year in range(min_date.year, max_date.year + 1):
        eos = pd.Timestamp(f"{year}-11-30")
        if min_date <= eos <= max_date:
            snap_dates.append(eos)
    # Add current snapshot (latest data)
    if max_date not in snap_dates:
        snap_dates.append(max_date)

    all_rows = []
    for snap in snap_dates:
        for tour in ["ATP", "WTA"]:
            t_obs = obs[obs["tour"] == tour]
            r = solve_tour(t_obs, snap)
            if r.empty:
                continue
            r["snapshot_date"] = snap
            r["tour"] = tour
            all_rows.append(r)
            print(f"  {tour} {snap.date()}: {len(r)} rated players")
    return pd.concat(all_rows, ignore_index=True)


def generate_data() -> None:
    """Read all_matches.csv, build observations, run rolling snapshots, write
    everything to docs/data/ following fleet conventions."""
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA / "players").mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(ALL_MATCHES_CSV)
    matches["date"] = pd.to_datetime(matches["date"])

    print("Building observations...")
    obs = build_observations(matches)
    print(f"  {len(obs):,} set-observations")

    print("\nBuilding rolling snapshots (year-end + current)...")
    ratings = build_rolling_snapshots(obs)
    print(f"\n{len(ratings):,} rating rows across {ratings['snapshot_date'].nunique()} snapshots")

    # Save full ratings CSV (gzipped — large)
    ratings_out = ratings.copy()
    ratings_out["snapshot_date"] = ratings_out["snapshot_date"].dt.strftime("%Y-%m-%d")
    ratings_out.to_csv(RATINGS_CSV, index=False, compression="gzip")
    print(f"Wrote {RATINGS_CSV}: {len(ratings_out):,} rows")

    # --- Current rankings (latest snapshot per tour) ---
    latest_snap = ratings["snapshot_date"].max()
    current = ratings[ratings["snapshot_date"] == latest_snap].copy()
    for tour in ["ATP", "WTA"]:
        t = current[current["tour"] == tour].sort_values("base", ascending=False).head(50).reset_index(drop=True)
        rows = []
        for i, r in t.iterrows():
            rows.append({
                "rank":         i + 1,
                "player":       r["player"],
                "base":         round(float(r["base"]), 3),
                "hard_delta":   round(float(r["hard_delta"]), 3),
                "clay_delta":   round(float(r["clay_delta"]), 3),
                "grass_delta":  round(float(r["grass_delta"]), 3),
                "carpet_delta": round(float(r["carpet_delta"]), 3),
                "sets_played":  int(r["sets_played"]),
            })
        out_path = DOCS_DATA / f"current_rankings_{tour.lower()}.json"
        with open(out_path, "w") as f:
            json.dump({
                "tour":      tour,
                "as_of":     str(latest_snap.date()),
                "players":   rows,
            }, f, separators=(",", ":"))
        print(f"  wrote {out_path.name} ({len(rows)} players)")

    # --- GOAT (peak base rating per player across all snapshots) ---
    for tour in ["ATP", "WTA"]:
        t = ratings[ratings["tour"] == tour].copy()
        # Find each player's peak snapshot (highest base rating)
        peaks = t.loc[t.groupby("player")["base"].idxmax()].copy()
        peaks = peaks.sort_values("base", ascending=False).head(50).reset_index(drop=True)
        rows = []
        for i, r in peaks.iterrows():
            rows.append({
                "rank":         i + 1,
                "player":       r["player"],
                "peak_snapshot": str(r["snapshot_date"].date()),
                "base":         round(float(r["base"]), 3),
                "hard_delta":   round(float(r["hard_delta"]), 3),
                "clay_delta":   round(float(r["clay_delta"]), 3),
                "grass_delta":  round(float(r["grass_delta"]), 3),
                "carpet_delta": round(float(r["carpet_delta"]), 3),
                "sets_played":  int(r["sets_played"]),
            })
        out_path = DOCS_DATA / f"goat_{tour.lower()}.json"
        with open(out_path, "w") as f:
            json.dump({
                "tour":     tour,
                "players":  rows,
            }, f, separators=(",", ":"))
        print(f"  wrote {out_path.name} ({len(rows)} players)")

    # --- Per-player history (all snapshots for each player) ---
    for tour in ["ATP", "WTA"]:
        t = ratings[ratings["tour"] == tour]
        for player, sub in t.groupby("player"):
            sub = sub.sort_values("snapshot_date")
            history = [
                {
                    "date":         str(r["snapshot_date"].date()),
                    "base":         round(float(r["base"]), 3),
                    "hard_delta":   round(float(r["hard_delta"]), 3),
                    "clay_delta":   round(float(r["clay_delta"]), 3),
                    "grass_delta":  round(float(r["grass_delta"]), 3),
                    "carpet_delta": round(float(r["carpet_delta"]), 3),
                    "sets_played":  int(r["sets_played"]),
                }
                for _, r in sub.iterrows()
            ]
            slug = _slug(player)
            with open(DOCS_DATA / "players" / f"{tour.lower()}_{slug}.json", "w") as f:
                json.dump({
                    "player":   player,
                    "tour":     tour,
                    "history":  history,
                }, f, separators=(",", ":"))
        print(f"  wrote per-player files for {tour}: {t['player'].nunique()} players")

    # --- Index of all rated players (for SPA dropdown) ---
    for tour in ["ATP", "WTA"]:
        t = ratings[ratings["tour"] == tour]
        index = []
        for player, sub in t.groupby("player"):
            peak = sub["base"].max()
            index.append({
                "name":     player,
                "tour":     tour,
                "slug":     _slug(player),
                "peak":     round(float(peak), 3),
            })
        index.sort(key=lambda r: r["name"])
        out_path = DOCS_DATA / f"players_index_{tour.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(index, f, separators=(",", ":"))
        print(f"  wrote {out_path.name} ({len(index)} players)")

    # --- Meta (date range, etc.) ---
    meta = {
        "first_match_date": str(matches["date"].min().date()),
        "last_match_date":  str(matches["date"].max().date()),
        "snapshots":        sorted(set(pd.Timestamp(d).strftime("%Y-%m-%d") for d in ratings["snapshot_date"].unique())),
        "generated_at":     datetime.utcnow().isoformat(),
    }
    with open(DOCS_DATA / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"  wrote meta.json")


def _slug(name: str) -> str:
    return re.sub(r"[^\w]", "_", name).strip("_")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    args = set(sys.argv[1:])
    if not args or "download" in args:
        print("=== Downloading Sackmann data ===")
        download_sackmann()
    if not args or "build" in args:
        print("\n=== Building unified all_matches.csv ===")
        build_unified()
    if "solve" in args or "all" in args:
        print("\n=== Building set-level observations ===")
        matches = pd.read_csv(ALL_MATCHES_CSV)
        matches["date"] = pd.to_datetime(matches["date"])
        obs = build_observations(matches)
        print(f"  {len(obs):,} set-observations built.")
        snapshot_date = matches["date"].max()
        for tour in ["ATP", "WTA"]:
            print(f"\n=== Solving {tour} ratings at {snapshot_date.date()} ===")
            ratings = solve_tour(obs[obs["tour"] == tour], snapshot_date)
            print(f"  Top 15 base ratings ({tour}):")
            print(ratings.head(15)[["player", "base", "hard_delta", "clay_delta",
                                    "grass_delta", "sets_played"]].to_string(index=False))
    if "generate" in args or "all" in args:
        print("\n=== Generating JSON snapshots for SPA ===")
        generate_data()
