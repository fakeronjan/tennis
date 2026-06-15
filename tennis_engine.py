"""
tennis_engine.py - fakeronjan WLS rating engine for tennis (singles, ATP + WTA, 1973+).

Pipeline:
  1. download_sackmann()   - fetch & cache Jeff Sackmann's tennis_atp + tennis_wta CSVs
  2. build_unified()       - parse + normalize into all_matches.csv (one row per match)
  3. build_observations()  - one row per SET, response = signed sqrt(game margin)
  4. solve_ratings()       - WLS solve: base rating + 4 surface deltas, per tour separately
  5. write_outputs()       - ratings CSVs + JSON snapshots for the SPA

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

# Tournament tier weights - mirrors MESSI's continental boost / ZIDANE's UCL boost.
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
    "A":  1.0,   # ATP 500/250 (we don't separate further - Sackmann doesn't always)
    "I":  0.8,   # WTA International (250-tier)
    "T2": 1.0,   # WTA T2
    "T3": 0.8,   # WTA T3
    "T4": 0.8,   # WTA T4
    "T5": 0.8,   # WTA T5
    "W":  1.0,   # Generic WTA Tour event (pre-2000, before T1..T5 split). Includes
                 # Virginia Slims Championships (1986-94) and WTA Tour Championships
                 # (1995-99) as YEC + the regular tour weeks.
    "D":  1.0,   # Davis Cup / Fed Cup (per user 2026-05-30)
    # Excluded: Challenger (C), Satellite (S), Futures, exhibition (E), country-vs-country
    # cup variants (CC)
}

# Surfaces. Indoor and outdoor hard are combined per industry convention.
# Carpet retired around 2009 but had a real role in '70s-'90s indoor events.
SURFACES = {"Hard", "Clay", "Grass", "Carpet"}


# ============================================================
# STEP 1 - DOWNLOAD SACKMANN
# ============================================================

USER_AGENT = "Mozilla/5.0 (compatible; fakeronjan tennis ratings bot)"


def _http_get(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _resolve_branch(repo: str, probe_file: str) -> "str | None":
    """Return whichever default branch the repo currently serves ('master' or
    'main'), or None if neither is reachable. GitHub doesn't redirect raw URLs
    when a repo renames its default branch, so we probe instead of assuming."""
    for br in ("master", "main"):
        url = f"https://raw.githubusercontent.com/{repo}/{br}/{probe_file}"
        try:
            _http_get(url, timeout=20)   # GET (HEAD is sometimes refused); small probe file
            return br
        except Exception:
            continue
    return None


def download_sackmann(min_year: int = MIN_YEAR, refresh_latest: int = 2) -> None:
    """Download (and cache) Jeff Sackmann's per-year ATP + WTA singles match CSVs.

    Idempotent: skips files already on disk except for the latest `refresh_latest`
    years (always re-fetched to absorb new matches mid-season).

    Auto-detects the upstream default branch (master vs main) so a rename doesn't
    silently break the build. Raises if nothing could be fetched.

    Layout written:
      data/sackmann/atp_matches_<YEAR>.csv
      data/sackmann/wta_matches_<YEAR>.csv
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    this_year = datetime.now().year
    refresh_from = this_year - refresh_latest + 1
    written = 0

    for tour, repo in [
        ("atp", "JeffSackmann/tennis_atp"),
        ("wta", "JeffSackmann/tennis_wta"),
    ]:
        branch = _resolve_branch(repo, f"{tour}_matches_{min_year}.csv")
        if branch is None:
            print(f"  ! {repo}: unreachable on master AND main - skipping")
            continue
        print(f"  {repo}: using '{branch}' branch")
        base = f"https://raw.githubusercontent.com/{repo}/{branch}"
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
                written += 1
                print(f"{len(data):,} bytes")
            except HTTPError as e:
                if e.code == 404:
                    print("(not yet posted)")
                else:
                    print(f"HTTP {e.code}")
            except Exception as e:
                print(f"FAILED ({e})")

    cached = len(list(DATA_DIR.glob("*.csv")))
    if written == 0 and cached == 0:
        raise RuntimeError(
            "download_sackmann fetched 0 files and no cache exists - Sackmann's "
            "tennis_atp / tennis_wta repos look unreachable or moved (checked both "
            "the master and main branches)."
        )


# ============================================================
# STEP 2 - BUILD UNIFIED all_matches.csv
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
            f"{DATA_DIR} not found - run download_sackmann() first."
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
        "winner_id", "winner_name", "winner_ioc",
        "loser_id", "loser_name", "loser_ioc",
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
# STEP 3 - PARSE SCORES + BUILD SET-LEVEL OBSERVATIONS
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
            # Skip nonsense (e.g. both 0 - already filtered, defensive)
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
# STEP 4 - WLS SOLVE (PER TOUR, SINGLE SNAPSHOT)
# ============================================================
# Each set is one observation. Each player gets 4 unknowns: a base rating +
# 3 surface deltas (hard/clay/grass). Carpet matches still feed the base
# rating but DO NOT get their own per-player delta - the modern tour has
# essentially zero carpet, so per-player carpet deltas over-fit on tiny
# samples and produced phantom +1.4 carpet bumps for players who'd played
# 2 carpet matches in a year. By absorbing carpet into base, we keep all
# the 1980s/90s match data without the noise.

SURFACE_LIST = ["Hard", "Clay", "Grass"]  # surfaces that get per-player deltas
PARAMS_PER_PLAYER = 1 + len(SURFACE_LIST)  # base + N deltas


MIN_SETS_PLAYED = 30  # filter for published rankings - drops tiny-sample noise

def solve_tour(obs: pd.DataFrame, snapshot_date: pd.Timestamp,
               window_days: int = 365,
               window_start: pd.Timestamp = None,
               recency_weighting: bool = True,
               min_sets_played: int = MIN_SETS_PLAYED) -> pd.DataFrame:
    """Run a single WLS solve on the window ending at snapshot_date.

    Returns a DataFrame: one row per player who meets min_sets_played, with columns
      [player, base, hard_delta, clay_delta, grass_delta, sets_played]

    Parameters:
      - snapshot_date: anchor date (matches up to and including this date enter).
      - window_days: width of the trailing rolling window in days (default 365).
        Ignored if `window_start` is given.
      - window_start: optional absolute start of the window. When supplied (e.g.
        Jan 1 of the calendar year for EOY snapshots), it overrides
        `snapshot_date - window_days`. Used to anchor calendar-year EOY snapshots
        that should not bleed into the prior year.
      - recency_weighting: when True (default), linear decay from 1.0 at the
        snapshot to ~0 at the window's far edge - the standard rolling-rating
        behavior. When False, every set in the window gets equal weight (still
        modulated by tier_weight). Used for EOY snapshots, where the snapshot
        should represent the year as a whole, not "form heading into Tour Finals".
      - min_sets_played: filter for published rankings.

    Implementation:
      - Filter obs to window [window_start, snapshot_date]
      - Apply recency decay (optional) + tier weights
      - Build X with 4 columns per player: base + 3 surface indicators (hard/clay/grass).
        Carpet matches contribute to base only (no surface column).
      - Solve via numpy.linalg.lstsq on weighted system
      - Anchor with two zero-sum constraints (cross-player base, per-player deltas)
      - Filter output to players with at least min_sets_played sets in window
    """
    snapshot_date = pd.Timestamp(snapshot_date)
    if window_start is None:
        window_start = snapshot_date - pd.Timedelta(days=window_days)
    else:
        window_start = pd.Timestamp(window_start)
    window_span_days = max(1, (snapshot_date - window_start).days)
    w = obs[(obs["date"] >= window_start) & (obs["date"] <= snapshot_date)].copy()
    if w.empty:
        return pd.DataFrame()

    if recency_weighting:
        # Linear recency decay: most-recent set weight = 1, oldest in window = ~0
        days_back = (snapshot_date - w["date"]).dt.days.to_numpy(dtype=float)
        recency = np.maximum(0.0, 1.0 - days_back / window_span_days)
    else:
        # Uniform weighting: every set in the window counts equally.
        recency = np.ones(len(w), dtype=float)
    sample_weight = w["tier_weight"].to_numpy(dtype=float) * recency

    # Build player index
    players = sorted(set(w["winner"]) | set(w["loser"]))
    pidx = {p: i for i, p in enumerate(players)}
    n_p = len(players)
    n_obs = len(w)
    if n_p < 4 or n_obs < 50:
        return pd.DataFrame()

    # Column layout: PARAMS_PER_PLAYER columns per player
    #   [base, hard_delta, clay_delta, grass_delta]
    # Carpet matches: contribute only to base (no surface column).
    SURF_COL = {s: i + 1 for i, s in enumerate(SURFACE_LIST)}  # +1 because base=0
    PPP = PARAMS_PER_PLAYER
    N_DELTAS = len(SURFACE_LIST)
    n_cols = n_p * PPP

    # Build X sparsely-ish (still dense numpy - for our 1-year window scale it fits)
    X = np.zeros((n_obs, n_cols), dtype=np.float32)
    y = np.zeros(n_obs, dtype=np.float32)
    sw = np.zeros(n_obs, dtype=np.float32)

    winners = w["winner"].to_numpy()
    losers = w["loser"].to_numpy()
    surfaces = w["surface"].to_numpy()
    y_obs = w["y"].to_numpy(dtype=np.float32)

    for i in range(n_obs):
        wi = pidx[winners[i]]
        li = pidx[losers[i]]
        # Base rating diff (always)
        X[i, wi * PPP] = 1.0
        X[i, li * PPP] = -1.0
        # Surface delta diff (only for hard/clay/grass; carpet goes to base only)
        s_col = SURF_COL.get(surfaces[i])
        if s_col is not None:
            X[i, wi * PPP + s_col] = 1.0
            X[i, li * PPP + s_col] = -1.0
        y[i] = y_obs[i]
        sw[i] = sample_weight[i]

    # Two anchoring constraints (very high weight, treated as hard constraints):
    #   1. Cross-player zero-sum on BASE - sum of all bases = 0 (anchors the
    #      overall rating scale).
    #   2. Per-player zero-sum on MODELED DELTAS - for each player,
    #      hard_delta + clay_delta + grass_delta = 0. This forces `base` to
    #      mean "average rating across the 3 modeled surfaces" rather than
    #      drifting with carpet performance. Without this constraint, players
    #      with carpet exposure see their base inflated by carpet matches
    #      (which only enter the model via base diff), producing GOAT-PEAK
    #      rankings dominated by carpet specialists.
    constraint_rows = 1 + n_p  # one cross-player + one-per-player
    X_con = np.zeros((constraint_rows, n_cols), dtype=np.float32)
    y_con = np.zeros(constraint_rows, dtype=np.float32)
    sw_con = np.full(constraint_rows, 1e7, dtype=np.float32)
    # Row 0: sum of bases = 0
    for p in range(n_p):
        X_con[0, p * PPP] = 1.0
    # Rows 1..n_p: per-player sum of modeled deltas = 0
    for p in range(n_p):
        for s in range(N_DELTAS):
            X_con[1 + p, p * PPP + 1 + s] = 1.0
    # Stash X/y/sw (we'll append ridge below and then the constraint rows on top)
    X = np.vstack([X, X_con])
    y = np.concatenate([y, y_con])
    sw = np.concatenate([sw, sw_con])

    # Ridge regularization on SURFACE DELTAS only (NOT base). This stabilizes
    # numerically singular snapshots where players have surface-imbalanced data
    # (e.g. AO snapshot date when many players have only played hard in the
    # window). Without this, lstsq finds extreme-magnitude solutions in the
    # null space. We DON'T regularize base ratings because we want them to
    # follow the data freely; the zero-sum constraint already anchors them.
    ridge_lambda = 0.01
    n_delta_rows = n_p * N_DELTAS
    X_ridge = np.zeros((n_delta_rows, n_cols), dtype=np.float32)
    y_ridge = np.zeros(n_delta_rows, dtype=np.float32)
    sw_ridge = np.full(n_delta_rows, ridge_lambda, dtype=np.float32)
    for p in range(n_p):
        for s in range(N_DELTAS):
            X_ridge[p * N_DELTAS + s, p * PPP + 1 + s] = 1.0
    X = np.vstack([X, X_ridge])
    y = np.concatenate([y, y_ridge])
    sw = np.concatenate([sw, sw_ridge])

    # Apply weights and solve
    sqrt_w = np.sqrt(sw)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    R = r.reshape(n_p, PPP)

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
        "sets_played":  [games_played.get(p, 0) for p in players],
    })
    # Filter to players with enough data in window
    out = out[out["sets_played"] >= min_sets_played].copy()
    return out.sort_values("base", ascending=False).reset_index(drop=True)


# ============================================================
# STEP 5 - ROLLING SNAPSHOTS + JSON OUTPUTS
# ============================================================
# Snapshot cadence: end-of-November each year (after ATP/WTA Finals usually) +
# current state. ~52 snapshots × 2 tours = ~100 solves, ~5-10 minutes total.

import csv
import json

DOCS_DATA = Path(__file__).parent / "docs" / "data"
RATINGS_CSV = Path(__file__).parent / "tennis_ratings.csv.gz"


SLAM_NAMES = {"Australian Open", "Roland Garros", "Wimbledon", "US Open", "Us Open"}
SLAM_TO_CODE = {
    "Australian Open": "AO",
    "Roland Garros":   "FO",
    "Wimbledon":       "Wim",
    "US Open":         "US",
    "Us Open":         "US",
}
SLAM_DISPLAY_ORDER = {"AO": 1, "FO": 2, "Wim": 3, "US": 4}


# IOC (3-letter Olympic) country code -> ISO-2 -> emoji flag. Covers every
# country that's ever produced an Open Era top-100 tennis player + historical
# entities (Czechoslovakia, USSR, FRG/GDR, Yugoslavia, etc.).
_IOC_TO_ISO2 = {
    "ARG": "AR", "ARM": "AM", "AUS": "AU", "AUT": "AT", "AZE": "AZ",
    "BAH": "BS", "BAR": "BB", "BEL": "BE", "BIH": "BA", "BLR": "BY",
    "BOL": "BO", "BRA": "BR", "BUL": "BG", "CAN": "CA", "CHI": "CL",
    "CHN": "CN", "COL": "CO", "CRC": "CR", "CRO": "HR", "CUB": "CU",
    "CYP": "CY", "CZE": "CZ", "DEN": "DK", "DOM": "DO", "ECU": "EC",
    "EGY": "EG", "ESA": "SV", "ESP": "ES", "EST": "EE", "FIN": "FI",
    "FRA": "FR", "FRG": "DE", "GBR": "GB", "GEO": "GE", "GER": "DE",
    "GDR": "DE", "GHA": "GH", "GRE": "GR", "GUA": "GT", "HAI": "HT",
    "HKG": "HK", "HON": "HN", "HUN": "HU", "INA": "ID", "IND": "IN",
    "IRI": "IR", "IRL": "IE", "ISL": "IS", "ISR": "IL", "ITA": "IT",
    "IVB": "VG", "JAM": "JM", "JPN": "JP", "KAZ": "KZ", "KEN": "KE",
    "KOR": "KR", "KSA": "SA", "KUW": "KW", "LAT": "LV", "LBN": "LB",
    "LIE": "LI", "LTU": "LT", "LUX": "LU", "MAR": "MA", "MAS": "MY",
    "MDA": "MD", "MEX": "MX", "MGL": "MN", "MKD": "MK", "MLT": "MT",
    "MNE": "ME", "MON": "MC", "NCA": "NI", "NED": "NL", "NGR": "NG",
    "NOR": "NO", "NZL": "NZ", "PAK": "PK", "PAN": "PA", "PAR": "PY",
    "PER": "PE", "PHI": "PH", "POL": "PL", "POR": "PT", "PUR": "PR",
    "QAT": "QA", "ROU": "RO", "ROM": "RO", "RSA": "ZA", "RTF": "RU",
    "RUS": "RU", "SCG": "RS", "SIN": "SG", "SLO": "SI", "SMR": "SM",
    "SRB": "RS", "SUI": "CH", "SUR": "SR", "SVK": "SK", "SWE": "SE",
    "TCH": "CZ", "THA": "TH", "TJK": "TJ", "TKM": "TM", "TPE": "TW",
    "TUN": "TN", "TUR": "TR", "UAE": "AE", "UKR": "UA", "URS": "RU",
    "URU": "UY", "USA": "US", "UZB": "UZ", "VEN": "VE", "VIE": "VN",
    "YUG": "RS", "ZIM": "ZW",
    # Lower-frequency IOC codes seen in our data
    "AHO": "NL", "ALG": "DZ", "ANG": "AO", "ANT": "AG", "BAN": "BD",
    "BEN": "BJ", "BER": "BM", "BHU": "BT", "BIZ": "BZ", "BOT": "BW",
    "BRU": "BN", "BUR": "BF", "CAY": "KY", "CGO": "CG", "CIV": "CI",
    "CMR": "CM", "ETH": "ET", "FIJ": "FJ", "GAM": "GM", "GUY": "GY",
    "HAW": "US", "ISV": "VI", "JOR": "JO", "KOS": "XK", "KSA": "SA",
    "KGZ": "KG", "LAO": "LA", "LCA": "LC", "LES": "LS", "LIB": "LB",
    "LBA": "LY", "MAD": "MG", "MAW": "MW", "MOZ": "MZ", "MRI": "MU",
    "MTN": "MR", "NAM": "NA", "NEP": "NP", "NGA": "NE", "NMI": "MP",
    "OMA": "OM", "PLE": "PS", "PNG": "PG", "PYF": "PF", "RWA": "RW",
    "SAM": "WS", "SEN": "SN", "SEY": "SC", "SGP": "SG", "SKN": "KN",
    "SLE": "SL", "SOL": "SB", "SOM": "SO", "SRI": "LK", "SUD": "SD",
    "SWZ": "SZ", "SYR": "SY", "TAN": "TZ", "TGA": "TO", "TOG": "TG",
    "TTO": "TT", "UGA": "UG", "VAN": "VU", "VIN": "VC", "YEM": "YE",
    "ZAM": "ZM",
}


def _iso2_to_flag(iso2):
    """Convert ISO-2 country code to regional-indicator emoji flag."""
    if not iso2 or len(iso2) != 2:
        return ""
    base = 0x1F1E6 - ord("A")
    return chr(base + ord(iso2[0])) + chr(base + ord(iso2[1]))


def country_flag(ioc):
    """Map a Sackmann IOC country code to a regional-indicator emoji flag.
    Returns empty string for unknown codes - the SPA renders that as no flag."""
    iso2 = _IOC_TO_ISO2.get(ioc)
    return _iso2_to_flag(iso2) if iso2 else ""


def build_player_countries(matches: pd.DataFrame) -> dict:
    """Per (tour, player) -> {ioc, flag} from the player's MOST RECENT match.
    Recent-over-bulk so defection cases (e.g. Navratilova TCH -> USA) reflect
    the player's late-career nationality rather than the bulk of their early
    appearances."""
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    parts = []
    for who, ioc in [("winner_name", "winner_ioc"), ("loser_name", "loser_ioc")]:
        sub = df[df[ioc].notna()][["tour", who, ioc, "date"]].rename(
            columns={who: "player", ioc: "ioc"})
        parts.append(sub)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(["tour", "player"], keep="last")
    return {
        (r["tour"], r["player"]): {"ioc": r["ioc"], "flag": country_flag(r["ioc"])}
        for _, r in combined.iterrows()
    }


def build_player_match_index(matches: pd.DataFrame) -> dict:
    """Pre-index every match by (tour, player_name) for fast window lookups.

    Each (tour, player) -> dict with parallel lists sorted by date:
      dates, won (bool), round, tourney
    Used by window_stats / career_stats helpers below.
    """
    out = {}
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    for tour in ["ATP", "WTA"]:
        sub = matches[matches["tour"] == tour].sort_values("date")
        for who_col, won in [("winner_name", True), ("loser_name", False)]:
            for player_name, grp in sub.groupby(who_col):
                key = (tour, player_name)
                if key not in out:
                    out[key] = {"dates": [], "won": [], "round": [], "tourney": []}
                b = out[key]
                b["dates"].extend(grp["date"].tolist())
                b["won"].extend([won] * len(grp))
                b["round"].extend(grp["round"].fillna("").tolist())
                b["tourney"].extend(grp["tourney_name"].fillna("").tolist())
    for key, b in out.items():
        order = sorted(range(len(b["dates"])), key=lambda i: b["dates"][i])
        for col in ("dates", "won", "round", "tourney"):
            b[col] = [b[col][i] for i in order]
    return out


def window_stats(per_player: dict, tour: str, player: str,
                 win_start: pd.Timestamp, win_end: pd.Timestamp):
    """Return (wins, losses, titles, slams_won_codes) for player in [win_start, win_end]."""
    bucket = per_player.get((tour, player))
    if not bucket:
        return 0, 0, 0, []
    wins = losses = titles = 0
    slam_codes = []
    for dt, won, rnd, tn in zip(bucket["dates"], bucket["won"],
                                  bucket["round"], bucket["tourney"]):
        if dt < win_start or dt > win_end:
            continue
        if won:
            wins += 1
            if rnd == "F":
                titles += 1
                if tn in SLAM_TO_CODE:
                    slam_codes.append(SLAM_TO_CODE[tn])
        else:
            losses += 1
    slams = sorted(set(slam_codes), key=lambda c: SLAM_DISPLAY_ORDER.get(c, 99))
    return wins, losses, titles, slams


def career_stats(per_player: dict, tour: str, player: str):
    """Return (career_wins, career_losses, career_titles, career_slams_count)
    across the player's entire match log - no window filtering."""
    bucket = per_player.get((tour, player))
    if not bucket:
        return 0, 0, 0, 0
    wins = losses = titles = slams = 0
    for won, rnd, tn in zip(bucket["won"], bucket["round"], bucket["tourney"]):
        if won:
            wins += 1
            if rnd == "F":
                titles += 1
                if tn in SLAM_TO_CODE:
                    slams += 1
        else:
            losses += 1
    return wins, losses, titles, slams


INCREMENTAL_STALENESS_DAYS = 60


def build_rolling_snapshots(obs: pd.DataFrame, matches: pd.DataFrame,
                            full_rebuild: bool = False,
                            cached_ratings: pd.DataFrame = None) -> pd.DataFrame:
    """Solve ratings at three kinds of anchor dates:

      - slam:    end date of each Grand Slam, with the standard 365-day rolling
                 window + recency decay (the "form heading into the slam" view).
      - eoy:     Dec 31 of each completed calendar year (or the year's last match
                 date if earlier), with a Jan 1 -> Dec 31 calendar-year window
                 and NO recency decay - the "power within this year" view.
                 Skipped for the current calendar year (in-progress).
      - current: the latest match date in the data (per tour). Standard rolling
                 window + recency decay. Only added if it differs from the latest
                 slam snapshot.

    Incremental mode (default for `generate`, overridden by --full):
      - If `cached_ratings` is provided and `full_rebuild` is False, snapshots
        whose date is older than INCREMENTAL_STALENESS_DAYS (60 days) AND that
        already exist in the cache are skipped and reused. Only the recent tail
        + brand-new (tour, date) anchors are solved.
      - Slam-end-date snapshots persist naturally in cache once solved.
      - The annual Dec 31 cron run uses --full to re-canvass everything against
        any retroactive Sackmann corrections.

    Sackmann stamps every match in a slam with the slam's start date, so a
    snapshot on that date naturally includes all the slam's matches in the
    rolling window. Per user 2026-05-30: snapshot anchor = actual slam date.

    Returns long-format DataFrame: one row per (snapshot_date, tour, player) with
    a `snapshot_type` column ("slam" / "eoy" / "current").
    """
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["year"] = matches["date"].dt.year

    slam_matches = matches[matches["tourney_name"].isin(SLAM_NAMES)]

    # Slam snapshots: one per (tour, slam-end-date).
    slam_pairs = set()
    for tour in ["ATP", "WTA"]:
        tour_slams = slam_matches[slam_matches["tour"] == tour]
        for d in sorted(tour_slams["date"].unique()):
            slam_pairs.add((tour, pd.Timestamp(d)))

    # Current snapshot per tour.
    current_pairs = set()
    for tour in ["ATP", "WTA"]:
        tour_latest = matches[matches["tour"] == tour]["date"].max()
        if pd.notna(tour_latest):
            current_pairs.add((tour, pd.Timestamp(tour_latest)))

    # EOY snapshots: per (tour, year), anchor on Dec 31 (or last match date that
    # year if earlier). Skip the current calendar year since EOY hasn't happened.
    eoy_pairs = set()
    current_year = datetime.now().year
    for tour in ["ATP", "WTA"]:
        tour_m = matches[matches["tour"] == tour]
        for yr in sorted(tour_m["year"].dropna().unique()):
            yr = int(yr)
            if yr >= current_year:
                continue  # in-progress year, no EOY yet
            year_matches = tour_m[tour_m["year"] == yr]
            if year_matches.empty:
                continue
            last_in_year = year_matches["date"].max()
            year_end = pd.Timestamp(year=yr, month=12, day=31)
            anchor = min(last_in_year, year_end)
            eoy_pairs.add((tour, pd.Timestamp(anchor), yr))

    # De-dup: avoid resolving (tour, date) twice if e.g. an EOY date happens to
    # equal a slam date (unlikely but defensive). slam takes precedence.
    eoy_by_pair = {(t, d): yr for (t, d, yr) in eoy_pairs}

    # Incremental mode: determine which snapshots can be reused from cache.
    # A snapshot is "cacheable" when its date is older than the staleness
    # window AND a copy already exists in cached_ratings for that (tour, date).
    cached_pairs = set()
    cached_rows_to_carry = None
    if not full_rebuild and cached_ratings is not None and not cached_ratings.empty:
        cached_ratings = cached_ratings.copy()
        cached_ratings["snapshot_date"] = pd.to_datetime(cached_ratings["snapshot_date"])
        staleness_cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=INCREMENTAL_STALENESS_DAYS)
        # Build the set of (tour, date) pairs already in cache
        cache_index = cached_ratings[["tour", "snapshot_date"]].drop_duplicates()
        for _, r in cache_index.iterrows():
            cached_pairs.add((r["tour"], pd.Timestamp(r["snapshot_date"])))
        # Snapshots we want to solve THIS run = the full queue
        wanted = set()
        for t, d in sorted(slam_pairs | current_pairs):
            wanted.add((t, d))
        for t, d, _yr in sorted(eoy_pairs):
            wanted.add((t, d))
        # Carry-over: in-cache pairs that are stale enough AND we still want them
        carry = [(t, d) for (t, d) in cached_pairs
                 if d < staleness_cutoff and (t, d) in wanted]
        cached_rows_to_carry = cached_ratings[
            cached_ratings.apply(
                lambda r: (r["tour"], pd.Timestamp(r["snapshot_date"])) in set(carry),
                axis=1,
            )
        ].copy()
        # Drop pairs we're carrying from the solve queue
        carry_set = set(carry)
        slam_pairs = {p for p in slam_pairs if p not in carry_set}
        current_pairs = {p for p in current_pairs if p not in carry_set}
        eoy_pairs = {(t, d, yr) for (t, d, yr) in eoy_pairs if (t, d) not in carry_set}
        print(f"  Incremental: reusing {len(carry_set)} cached snapshots "
              f"(stale > {INCREMENTAL_STALENESS_DAYS} days)")

    print(f"  {len(slam_pairs)} slam + {len(eoy_pairs)} EOY + "
          f"{len(current_pairs - slam_pairs)} current-only snapshots to solve")

    all_rows = []
    last_print_year = None

    def _add(tour, snap, snap_type, window_start=None, recency_weighting=True):
        t_obs = obs[obs["tour"] == tour]
        r = solve_tour(t_obs, snap,
                       window_start=window_start,
                       recency_weighting=recency_weighting)
        if r.empty:
            return
        r["snapshot_date"] = snap
        r["tour"] = tour
        r["snapshot_type"] = snap_type
        all_rows.append(r)

    for tour, snap in sorted(slam_pairs | current_pairs):
        snap_type = "slam" if (tour, snap) in slam_pairs else "current"
        _add(tour, snap, snap_type)
        yr = snap.year
        if yr != last_print_year:
            print(f"  ... year {yr}")
            last_print_year = yr

    for tour, snap, yr in sorted(eoy_pairs):
        window_start = pd.Timestamp(year=yr, month=1, day=1)
        _add(tour, snap, "eoy", window_start=window_start, recency_weighting=False)

    fresh = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if cached_rows_to_carry is not None and not cached_rows_to_carry.empty:
        return pd.concat([cached_rows_to_carry, fresh], ignore_index=True)
    return fresh


def generate_data(full_rebuild: bool = False) -> None:
    """Read all_matches.csv, build observations, run rolling snapshots, write
    everything to docs/data/ following fleet conventions.

    Incremental by default (re-solves only the last 60 days of snapshots, reuses
    cached older ones). Pass `full_rebuild=True` to force a full re-solve."""
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA / "players").mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(ALL_MATCHES_CSV)
    matches["date"] = pd.to_datetime(matches["date"])

    # Per-player country (most recent nationality from match data).
    player_country = build_player_countries(matches)

    print("Building observations...")
    obs = build_observations(matches)
    print(f"  {len(obs):,} set-observations")

    print("\nBuilding slam-day snapshots + current...")
    # Load cached ratings (if any) - drives the incremental skip logic in
    # build_rolling_snapshots. Ignored under full_rebuild.
    cached_ratings = None
    if not full_rebuild and RATINGS_CSV.exists():
        cached_ratings = pd.read_csv(RATINGS_CSV)
        print(f"\nLoaded {len(cached_ratings):,} cached rating rows for incremental skip.")
    ratings = build_rolling_snapshots(obs, matches,
                                       full_rebuild=full_rebuild,
                                       cached_ratings=cached_ratings)
    print(f"\n{len(ratings):,} rating rows across {ratings['snapshot_date'].nunique()} snapshots")

    # Save full ratings CSV (gzipped - large)
    ratings_out = ratings.copy()
    ratings_out["snapshot_date"] = ratings_out["snapshot_date"].dt.strftime("%Y-%m-%d")
    ratings_out.to_csv(RATINGS_CSV, index=False, compression="gzip")
    print(f"Wrote {RATINGS_CSV}: {len(ratings_out):,} rows")

    # --- Power Rankings history: ALL snapshots (slam + eoy + current) per tour ---
    # Powers the Year + Within-Year picker on the Power Rankings tab.
    write_power_rankings_history(ratings, matches)

    # --- Current rankings (latest snapshot PER TOUR - ATP and WTA can differ) ---
    # Eligibility gates differ by snapshot type AND by tour. WTA matches max
    # at best-of-3, ATP slams at best-of-5, so women naturally play fewer
    # sets per match - and ~78% the sets per year vs ATP top-30 medians
    # (WTA ~121, ATP ~156). Tour-specific thresholds keep gate strictness
    # constant relative to a full season's natural sample.
    GATES = {
        # tour -> (rolling 1y minimum, EOY/GOAT minimum)
        "ATP": (80, 125),
        "WTA": (64, 100),
    }
    for tour in ["ATP", "WTA"]:
        rolling_min, _ = GATES[tour]
        tour_ratings = ratings[ratings["tour"] == tour]
        latest_snap = tour_ratings["snapshot_date"].max()
        latest = tour_ratings[tour_ratings["snapshot_date"] == latest_snap]
        latest = latest[latest["sets_played"] >= rolling_min]
        t = latest.sort_values("base", ascending=False).head(50).reset_index(drop=True)
        rows = []
        for i, r in t.iterrows():
            cinfo = player_country.get((tour, r["player"]), {})
            rows.append({
                "rank":         i + 1,
                "player":       r["player"],
                "country":      cinfo.get("flag", ""),
                "country_ioc":  cinfo.get("ioc", ""),
                "base":         round(float(r["base"]), 3),
                "hard_delta":   round(float(r["hard_delta"]), 3),
                "clay_delta":   round(float(r["clay_delta"]), 3),
                "grass_delta":  round(float(r["grass_delta"]), 3),
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

    # --- GOAT-PEAK + GOAT-ERA (EOY snapshots only) ---
    # Both views use the EOY (end-of-year) snapshot - calendar-year window,
    # tier weighting only, no recency decay. One full-season-calibrated
    # rating per (player, year).
    #   PEAK = player's max EOY base rating across career. Surfaces the
    #          peak-year body of work (W-L, titles, slams that year).
    #   ERA  = sum of (player's positive EOY base ratings) across career.
    #          Surfaces career totals (slams, titles, year-end #1 finishes).
    #          Negative years clipped at 0 - body of work above tour average.
    eoy_only = ratings[ratings["snapshot_type"] == "eoy"].copy()
    eoy_only["year"] = pd.to_datetime(eoy_only["snapshot_date"]).dt.year

    # Shared per-player match log for window/career stat lookups.
    per_player = build_player_match_index(matches)

    # Tour-specific EOY-set gate for the calendar-year GOAT views (PEAK /
    # All Seasons / ERA). WTA matches max at best-of-3 vs ATP slams' best-
    # of-5, so women naturally play ~78% the sets per year of ATP top-30
    # equivalents (WTA median 121, ATP median 156). Same threshold for both
    # tours would punish Serena and other WTA legends who scheduled
    # selectively but still beat the field.
    EOY_GATES = {"ATP": 125, "WTA": 100}

    def qualifies(row):
        return row["sets_played"] >= EOY_GATES.get(row["tour"], 125)

    eoy_peak = eoy_only[eoy_only.apply(qualifies, axis=1)].copy()
    eoy_era  = eoy_peak
    print(f"  GOAT eligibility filter (ATP >={EOY_GATES['ATP']}, WTA >={EOY_GATES['WTA']}): "
          f"{len(eoy_only):,} -> {len(eoy_peak):,} player-years")

    # Per-year anchor: rating of the 10th-ranked player that year. Adjusting
    # by this anchor turns each year's PEAK rating into "wins above the year's
    # marginal top-10 player" - strips out field-depth and tournament-coverage
    # variance across years, and surfaces the truly elite peaks. Side effect
    # for ERA: a year only contributes to a player's career ERA if they
    # cleared the year's top-10 bar. ERA becomes a "years of elite tennis"
    # quality-weighted measure.
    ANCHOR_RANK = 10
    anchor_per_year = {}
    for (tour, yr), grp in eoy_only.groupby(["tour", "year"]):
        sorted_g = grp.sort_values("base", ascending=False).reset_index(drop=True)
        if len(sorted_g) >= ANCHOR_RANK:
            anchor_per_year[(tour, int(yr))] = float(sorted_g.iloc[ANCHOR_RANK - 1]["base"])
        else:
            # If a year has fewer EOY-published players than the anchor rank
            # (very early WTA), use the last player as the anchor.
            anchor_per_year[(tour, int(yr))] = float(sorted_g.iloc[-1]["base"])

    def _adjust(df):
        df["anchor_50"] = df.apply(
            lambda r: anchor_per_year.get((r["tour"], int(r["year"])), 0.0), axis=1)
        df["adj_base"] = df["base"] - df["anchor_50"]
        return df
    eoy_peak = _adjust(eoy_peak)
    eoy_era  = _adjust(eoy_era)

    for tour in ["ATP", "WTA"]:
        t_peak = eoy_peak[eoy_peak["tour"] == tour].copy()
        t_era  = eoy_era[eoy_era["tour"] == tour].copy()
        if t_peak.empty and t_era.empty:
            continue

        # Year-end #1 count per player - uses the looser ERA filter so the
        # tally credits dominant-but-thin years (e.g. Hingis 1997 first WTA YE#1).
        no1_per_year = t_era.loc[t_era.groupby("year")["base"].idxmax()]
        no1_counts = no1_per_year.groupby("player").size().to_dict()

        # ----- PEAK -----
        # Rank by adj_base (rating above year's #50 player) to make cross-year
        # comparison meaningful. Display the adjusted value as the headline rating.
        peaks = t_peak.loc[t_peak.groupby("player")["adj_base"].idxmax()].copy()
        peaks = peaks.sort_values("adj_base", ascending=False).head(50).reset_index(drop=True)
        peak_rows = []
        for i, r in peaks.iterrows():
            year = int(r["year"])
            yr_start = pd.Timestamp(year=year, month=1, day=1)
            yr_end   = pd.Timestamp(year=year, month=12, day=31)
            wins, losses, titles, slams = window_stats(
                per_player, tour, r["player"], yr_start, yr_end)
            cinfo = player_country.get((tour, r["player"]), {})
            peak_rows.append({
                "rank":          i + 1,
                "player":        r["player"],
                "country":       cinfo.get("flag", ""),
                "country_ioc":   cinfo.get("ioc", ""),
                "peak_year":     year,
                "base":          round(float(r["adj_base"]), 3),   # adjusted headline value
                "raw_base":      round(float(r["base"]), 3),
                "anchor_50":     round(float(r["anchor_50"]), 3),
                "hard_delta":    round(float(r["hard_delta"]), 3),
                "clay_delta":    round(float(r["clay_delta"]), 3),
                "grass_delta":   round(float(r["grass_delta"]), 3),
                "match_wins":    wins,
                "match_losses":  losses,
                "titles":        titles,
                "slams_won":     slams,
                "year_end_no1":  int(no1_counts.get(r["player"], 0)),
            })
        with open(DOCS_DATA / f"goat_peak_{tour.lower()}.json", "w") as f:
            json.dump({"tour": tour, "view": "PEAK", "players": peak_rows},
                      f, separators=(",", ":"))
        print(f"  wrote goat_peak_{tour.lower()}.json ({len(peak_rows)} players)")

        # ----- PEAK (All seasons) -----
        # Top 50 (player, year) combinations by adjusted base. Same player
        # can appear multiple times if they had multiple elite seasons -
        # complements the "Best per player" view by surfacing the era-
        # dominant stretches (e.g. Federer 2005/06/07 or Nadal 2008/10/13).
        all_seasons = t_peak.sort_values("adj_base", ascending=False).head(50).reset_index(drop=True)
        all_rows = []
        for i, r in all_seasons.iterrows():
            year = int(r["year"])
            yr_start = pd.Timestamp(year=year, month=1, day=1)
            yr_end   = pd.Timestamp(year=year, month=12, day=31)
            wins, losses, titles, slams = window_stats(
                per_player, tour, r["player"], yr_start, yr_end)
            cinfo = player_country.get((tour, r["player"]), {})
            all_rows.append({
                "rank":          i + 1,
                "player":        r["player"],
                "country":       cinfo.get("flag", ""),
                "country_ioc":   cinfo.get("ioc", ""),
                "peak_year":     year,
                "base":          round(float(r["adj_base"]), 3),
                "raw_base":      round(float(r["base"]), 3),
                "anchor_50":     round(float(r["anchor_50"]), 3),
                "hard_delta":    round(float(r["hard_delta"]), 3),
                "clay_delta":    round(float(r["clay_delta"]), 3),
                "grass_delta":   round(float(r["grass_delta"]), 3),
                "match_wins":    wins,
                "match_losses":  losses,
                "titles":        titles,
                "slams_won":     slams,
                "year_end_no1":  int(no1_counts.get(r["player"], 0)),
            })
        with open(DOCS_DATA / f"goat_peak_seasons_{tour.lower()}.json", "w") as f:
            json.dump({"tour": tour, "view": "PEAK_SEASONS", "players": all_rows},
                      f, separators=(",", ":"))
        print(f"  wrote goat_peak_seasons_{tour.lower()}.json ({len(all_rows)} player-seasons)")

        # ----- ERA -----
        # Sum of positive ADJUSTED EOY ratings across each player's career.
        # Adjusted = base - year's #50-player base. Negative adjusted years
        # (player below the year's marginal tour participant) contribute 0.
        t_era["positive"] = t_era["adj_base"].clip(lower=0)
        era_score = t_era.groupby("player").agg(
            era=("positive", "sum"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            years_active=("year", "nunique"),
        ).reset_index()
        era_score = era_score.sort_values("era", ascending=False).head(50).reset_index(drop=True)

        era_rows = []
        for i, r in era_score.iterrows():
            # Career totals across the player's entire match log (no window).
            c_wins, c_losses, c_titles, c_slams = career_stats(
                per_player, tour, r["player"])
            cinfo = player_country.get((tour, r["player"]), {})
            era_rows.append({
                "rank":             i + 1,
                "player":           r["player"],
                "country":          cinfo.get("flag", ""),
                "country_ioc":      cinfo.get("ioc", ""),
                "era":              round(float(r["era"]), 3),
                "first_year":       int(r["first_year"]),
                "last_year":        int(r["last_year"]),
                "years_active":     int(r["years_active"]),
                "career_wins":      c_wins,
                "career_losses":    c_losses,
                "career_titles":    c_titles,
                "career_slams":     c_slams,
                "year_end_no1":     int(no1_counts.get(r["player"], 0)),
            })
        with open(DOCS_DATA / f"goat_era_{tour.lower()}.json", "w") as f:
            json.dump({"tour": tour, "view": "ERA", "players": era_rows},
                      f, separators=(",", ":"))
        print(f"  wrote goat_era_{tour.lower()}.json ({len(era_rows)} players)")

    # --- Per-player history (all snapshots for each player) ---
    # Pre-compute rank per (tour, snapshot_date, player) - within the
    # tour- and type-appropriate qualifying top 50:
    #   ATP EOY: >=125  ATP rolling: >=80
    #   WTA EOY: >=100  WTA rolling: >=64
    # Players outside that top 50 get rank=null so the SPA renders a hyphen.
    GATE_LOOKUP = {  # (tour, is_eoy) -> min_sets
        ("ATP", True):  125, ("ATP", False): 80,
        ("WTA", True):  100, ("WTA", False): 64,
    }
    rank_lookup = {}  # (tour, date, player) -> rank (1..50) or None
    for (tour_x, snap_dt), grp in ratings.groupby(["tour", "snapshot_date"]):
        snap_type = grp["snapshot_type"].iloc[0]
        min_sets = GATE_LOOKUP[(tour_x, snap_type == "eoy")]
        qualifying = grp[grp["sets_played"] >= min_sets]
        sorted_grp = qualifying.sort_values("base", ascending=False).reset_index(drop=True)
        for i, row in sorted_grp.iterrows():
            r = i + 1
            if r <= 50:
                rank_lookup[(tour_x, pd.Timestamp(snap_dt), row["player"])] = r

    for tour in ["ATP", "WTA"]:
        t = ratings[ratings["tour"] == tour]
        for player, sub in t.groupby("player"):
            sub = sub.sort_values("snapshot_date")
            history = []
            years_in_history = set()
            for _, r in sub.iterrows():
                snap = pd.Timestamp(r["snapshot_date"])
                rank = rank_lookup.get((tour, snap, player))
                history.append({
                    "date":         str(snap.date()),
                    "base":         round(float(r["base"]), 3),
                    "hard_delta":   round(float(r["hard_delta"]), 3),
                    "clay_delta":   round(float(r["clay_delta"]), 3),
                    "grass_delta":  round(float(r["grass_delta"]), 3),
                    "sets_played":  int(r["sets_played"]),
                    "rank":         rank,
                    "snapshot_type": r["snapshot_type"],
                })
                years_in_history.add(snap.year)
            # Per-year aggregate stats (record, titles, slams won) - used by
            # the Player Summary slam-grid Record/Titles/EOY columns.
            year_stats = {}
            for yr in sorted(years_in_history):
                ys = pd.Timestamp(year=int(yr), month=1, day=1)
                ye = pd.Timestamp(year=int(yr), month=12, day=31)
                w, l, ti, sl = window_stats(per_player, tour, player, ys, ye)
                year_stats[str(int(yr))] = {
                    "wins":   w, "losses": l, "titles": ti, "slams_won": sl,
                }
            slug = _slug(player)
            cinfo = player_country.get((tour, player), {})
            with open(DOCS_DATA / "players" / f"{tour.lower()}_{slug}.json", "w") as f:
                json.dump({
                    "player":      player,
                    "tour":        tour,
                    "country":     cinfo.get("flag", ""),
                    "country_ioc": cinfo.get("ioc", ""),
                    "history":     history,
                    "year_stats":  year_stats,
                }, f, separators=(",", ":"))
        print(f"  wrote per-player files for {tour}: {t['player'].nunique()} players")

    # --- Index of all rated players (for SPA dropdown) ---
    for tour in ["ATP", "WTA"]:
        t = ratings[ratings["tour"] == tour]
        index = []
        for player, sub in t.groupby("player"):
            peak = sub["base"].max()
            cinfo = player_country.get((tour, player), {})
            index.append({
                "name":         player,
                "tour":         tour,
                "slug":         _slug(player),
                "country":      cinfo.get("flag", ""),
                "country_ioc":  cinfo.get("ioc", ""),
                "peak":         round(float(peak), 3),
            })
        index.sort(key=lambda r: r["name"])
        out_path = DOCS_DATA / f"players_index_{tour.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(index, f, separators=(",", ":"))
        print(f"  wrote {out_path.name} ({len(index)} players)")

    # --- Champions: append any newly-completed slam from match data, then overlay ---
    # update_slam_champions extends the curated slams_{m,w}.csv + slams.json with new
    # finals (keeps the Champions tab self-updating without a Wikipedia scrape).
    # write_champion_ratings then reads the (now-current) CSVs to build the rating
    # overlay keyed by `{tourLetter}_{year}_{spaCode}`.
    update_slam_champions(matches)
    write_champion_ratings(matches, ratings)

    # --- Meta (date range, etc.) ---
    # Whether the latest match date is a Grand Slam start: the SPA/portal nudge the
    # coverage end-date +2 weeks only for slams (which run ~2 weeks past their stamped
    # start), so a between-slams 1-week event doesn't get a spurious month bump.
    last_date = matches["date"].max()
    last_is_slam = bool(((matches["date"] == last_date) &
                         (matches["tourney_name"].isin(SLAM_NAMES))).any())
    meta = {
        "first_match_date":   str(matches["date"].min().date()),
        "last_match_date":    str(last_date.date()),
        "last_match_is_slam": last_is_slam,
        "snapshots":          sorted(set(pd.Timestamp(d).strftime("%Y-%m-%d") for d in ratings["snapshot_date"].unique())),
        "generated_at":       datetime.utcnow().isoformat(),
    }
    with open(DOCS_DATA / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"  wrote meta.json")


def write_power_rankings_history(ratings: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Emit docs/data/power_rankings_history_{atp,wta}.json - every snapshot the
    SPA needs for the Year + Within-Year picker on the Power Rankings tab.

    Snapshot keys:
      - "{year}_AO"   - after Australian Open
      - "{year}_FO"   - after Roland Garros
      - "{year}_Wim"  - after Wimbledon
      - "{year}_US"   - after US Open
      - "{year}_EOY"  - Dec 31 calendar-year snapshot (full year, no recency)
      - "Today"       - latest data point (only for current calendar year)

    Each snapshot includes per-player rating stats (base + surface deltas) PLUS
    window-scoped career stats: match W-L, title count, slam codes won. The
    SPA renders these in additional columns on the Power Rankings table.
    """
    SLAM_LABELS = {"AO": "After Australian Open", "FO": "After Roland Garros",
                   "Wim": "After Wimbledon", "US": "After US Open",
                   "EOY": "End of year (full year, no recency)",
                   "Today": "Today (latest data)"}

    # COVID-disrupted snapshots: rolling windows that cross the Mar-Aug 2020 tour
    # shutdown or that contain irregular slam-calendar configurations. Format
    # matches the fleet-wide disrupted-season pattern (tag, category, note).
    # Only the "covid" category applies here - tennis had no work stoppage or
    # cancelled-season equivalents to the labor / cancelled tags used elsewhere.
    DISRUPTED_SNAPSHOTS = {
        "2020_FO":  {"tag": "covid", "category": "covid",
                     "note": "Roland Garros 2020 rescheduled to Sept-Oct (normally late May/early June) after the Mar-Aug COVID tour shutdown."},
        "2020_US":  {"tag": "covid", "category": "covid",
                     "note": "US Open 2020 played in the NYC bubble with a thin field - Federer, Nadal, and several other top players sat out; Djokovic was defaulted."},
        "2021_AO":  {"tag": "covid", "category": "covid",
                     "note": "Australian Open 2021 delayed to February. The 365-day rolling window reaches into the Mar-Aug 2020 tour shutdown."},
        "2021_FO":  {"tag": "covid", "category": "covid",
                     "note": "365-day rolling window contains TWO Roland Garros editions (Sept 2020 + Jun 2021) and no Wimbledon (2020 cancelled)."},
        "2021_Wim": {"tag": "covid", "category": "covid",
                     "note": "First Wimbledon since 2019 (2020 cancelled). Window reaches into a year without a played Wimbledon."},
        "2021_US":  {"tag": "covid", "category": "covid",
                     "note": "365-day rolling window still contains TWO Roland Garros editions (Sept 2020 + Jun 2021)."},
    }

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])

    # Build (tour, snapshot_date) -> slam_code lookup from the matches data.
    slam_code_lookup = {}
    for code, name_set in [("AO", {"Australian Open"}),
                            ("FO", {"Roland Garros"}),
                            ("Wim", {"Wimbledon"}),
                            ("US", {"US Open", "Us Open"})]:
        sub = matches[matches["tourney_name"].isin(name_set)]
        for tour in ["ATP", "WTA"]:
            for d in sub[sub["tour"] == tour]["date"].unique():
                slam_code_lookup[(tour, pd.Timestamp(d))] = code

    per_player = build_player_match_index(matches)
    player_country = build_player_countries(matches)

    for tour in ["ATP", "WTA"]:
        t = ratings[ratings["tour"] == tour].copy()
        if t.empty:
            continue
        snapshots = {}
        current_rows = t[t["snapshot_type"] == "current"]
        today_date = current_rows["snapshot_date"].max() if not current_rows.empty else None

        for snap_date, group in t.groupby("snapshot_date"):
            snap_date = pd.Timestamp(snap_date)
            row_type = group["snapshot_type"].iloc[0]
            year = snap_date.year
            if row_type == "slam":
                code = slam_code_lookup.get((tour, snap_date))
                if not code:
                    continue
                key = f"{year}_{code}"
                # 365-day trailing window for slam snapshots
                win_start = snap_date - pd.Timedelta(days=365)
                win_end   = snap_date
            elif row_type == "eoy":
                code = "EOY"
                key = f"{year}_EOY"
                # Calendar year window for EOY snapshots
                win_start = pd.Timestamp(year=int(year), month=1, day=1)
                win_end   = snap_date  # the actual EOY anchor
            elif row_type == "current":
                if today_date is None or snap_date != today_date:
                    continue
                code = "Today"
                key = "Today"
                win_start = snap_date - pd.Timedelta(days=365)
                win_end   = snap_date
            else:
                continue
            # Snapshot-type AND tour-aware gate (matches the rank_lookup above
            # and the per-tour calibration in current_rankings + GOAT).
            gate = {("ATP", True): 125, ("ATP", False): 80,
                    ("WTA", True): 100, ("WTA", False): 64}
            min_sets = gate[(tour, row_type == "eoy")]
            qualifying = group[group["sets_played"] >= min_sets]
            top = qualifying.sort_values("base", ascending=False).head(50).reset_index(drop=True)
            players = []
            for i, r in top.iterrows():
                wins, losses, titles, slams = window_stats(
                    per_player, tour, r["player"], win_start, win_end)
                cinfo = player_country.get((tour, r["player"]), {})
                players.append({
                    "rank":         i + 1,
                    "player":       r["player"],
                    "country":      cinfo.get("flag", ""),
                    "country_ioc":  cinfo.get("ioc", ""),
                    "base":         round(float(r["base"]), 3),
                    "hard_delta":   round(float(r["hard_delta"]), 3),
                    "clay_delta":   round(float(r["clay_delta"]), 3),
                    "grass_delta":  round(float(r["grass_delta"]), 3),
                    "sets_played":  int(r["sets_played"]),
                    "match_wins":   wins,
                    "match_losses": losses,
                    "titles":       titles,
                    "slams_won":    slams,
                })
            snap_entry = {
                "year":         int(year) if code != "Today" else int(year),
                "code":         code,
                "label":        SLAM_LABELS.get(code, code),
                "date":         str(snap_date.date()),
                "window_start": str(win_start.date()),
                "window_end":   str(win_end.date()),
                "type":         row_type,
                "players":      players,
            }
            disrupted = DISRUPTED_SNAPSHOTS.get(key)
            if disrupted:
                snap_entry["disrupted"] = disrupted
            snapshots[key] = snap_entry
        out = {
            "tour":         tour,
            "snapshots":    snapshots,
        }
        path = DOCS_DATA / f"power_rankings_history_{tour.lower()}.json"
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"  wrote {path.name} ({len(snapshots)} snapshots)")


# IOC (3-letter, Sackmann winner_ioc) -> ISO2, for the slams.json player-flag map
# that the SPA Champions grid renders via flagEmoji(). Covers every nation that has
# produced an Open Era Grand Slam singles champion plus likely near-future ones.
# update_slam_champions raises on any code missing here, so a new champion's blank
# flag fails the run loudly rather than shipping silently.
IOC_TO_ISO2 = {
    "USA": "US", "ESP": "ES", "SUI": "CH", "SRB": "RS", "GER": "DE", "FRG": "DE",
    "GDR": "DE", "SWE": "SE", "AUS": "AU", "GBR": "GB", "RUS": "RU", "ITA": "IT",
    "ARG": "AR", "BRA": "BR", "ROU": "RO", "AUT": "AT", "CRO": "HR", "RSA": "ZA",
    "FRA": "FR", "ECU": "EC", "NED": "NL", "CZE": "CZ", "TCH": "CZ", "BEL": "BE",
    "JPN": "JP", "POL": "PL", "CHN": "CN", "BLR": "BY", "KAZ": "KZ", "LAT": "LV",
    "DEN": "DK", "CAN": "CA", "YUG": "RS", "URS": "RU", "MEX": "MX", "BUL": "BG",
    "NOR": "NO", "GRE": "GR", "SVK": "SK", "TUN": "TN", "POR": "PT",
}


def update_slam_champions(matches: pd.DataFrame) -> None:
    """Append newly-completed Grand Slam champions found in the match data to the
    curated slams_{m,w}.csv lists, then rebundle slams.json.

    The historical champion list (slams_{m,w}.csv) is curated and authoritative:
    Sackmann's match data is missing every 1968-1972 final and mis-aligns pre-1987
    Australian Open years (the AO was played in December then, and Sackmann's
    tournament-date stamping disagrees with the calendar-year convention the site
    uses). So we never re-derive history - we only APPEND slams that completed after
    the curated list was last extended, taking each champion from that slam's Final
    ('F' round) winner. That keeps the Champions tab self-updating from match data
    with no Wikipedia scrape in the daily cron. CSV slam codes are AO/RG/Wim/US;
    slams.json uses FO for RG (matching the existing bundle)."""
    root = Path(__file__).parent
    name_to_csv_code = {"Australian Open": "AO", "Roland Garros": "RG",
                        "Wimbledon": "Wim", "US Open": "US", "Us Open": "US"}
    slam_order = {"AO": 0, "RG": 1, "Wim": 2, "US": 3}

    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["year"] = m["date"].dt.year
    finals = m[m["round"] == "F"].copy()
    finals["csv_code"] = finals["tourney_name"].map(name_to_csv_code)
    finals = finals[finals["csv_code"].notna()]

    added = {"m": [], "w": []}
    for tour_letter, tour in (("m", "ATP"), ("w", "WTA")):
        csv_path = root / f"slams_{tour_letter}.csv"
        if not csv_path.exists():
            print(f"  skipping slam-champion update (slams_{tour_letter}.csv not found)")
            return
        s = pd.read_csv(csv_path)
        existing = set(zip(s["year"].astype(int), s["slam"]))
        next_num = int(s["slam_num"].max()) + 1 if len(s) else 1
        new_rows = []
        for _, r in finals[finals["tour"] == tour].iterrows():
            key = (int(r["year"]), r["csv_code"])
            if key not in existing:
                new_rows.append((key[0], key[1], r["winner_name"], r["winner_ioc"]))
        # One Final per (year, code); sort chronologically before appending.
        new_rows = sorted(set(new_rows), key=lambda x: (x[0], slam_order.get(x[1], 9)))
        if not new_rows:
            print(f"  slams_{tour_letter}.csv: no new champions")
            continue
        with open(csv_path, "a", newline="") as f:
            wr = csv.writer(f)
            for yr, code, name, _ioc in new_rows:
                wr.writerow([yr, code, name, "", next_num])  # country col unused by overlay
                next_num += 1
        added[tour_letter] = new_rows
        print(f"  slams_{tour_letter}.csv: +{len(new_rows)} (" +
              ", ".join(f"{y} {c} {n}" for y, c, n, _ in new_rows) + ")")

    if not added["m"] and not added["w"]:
        return
    bundle_path = root / "slams.json"
    bundle = json.load(open(bundle_path))
    csv_to_spa = {"RG": "FO"}
    for tour_letter in ("m", "w"):
        for yr, code, name, ioc in added[tour_letter]:
            bundle["data"][tour_letter].append(
                {"y": yr, "s": csv_to_spa.get(code, code), "w": name})
            if name not in bundle["players"][tour_letter]:
                iso = IOC_TO_ISO2.get(ioc)
                if not iso:
                    raise SystemExit(
                        f"update_slam_champions: no ISO2 for IOC {ioc!r} (champion "
                        f"{name}); add it to IOC_TO_ISO2")
                bundle["players"][tour_letter][name] = iso
    json.dump(bundle, open(bundle_path, "w"), ensure_ascii=False)
    print(f"  rebundled slams.json (+{len(added['m']) + len(added['w'])} champions)")


def write_champion_ratings(matches: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Emit docs/data/champion_ratings.json - used by the Champions tab to show
    each Grand Slam winner's base + surface-delta at the time they won."""
    SLAM_NAME    = {"AO": "Australian Open", "FO": "Roland Garros", "Wim": "Wimbledon", "US": "US Open"}
    SLAM_SURFACE = {"AO": "hard", "FO": "clay", "Wim": "grass", "US": "hard"}
    CSV_CODE_MAP = {"RG": "FO"}  # slams_*.csv uses RG; SPA bundle uses FO

    slams_m_csv = Path(__file__).parent / "slams_m.csv"
    slams_w_csv = Path(__file__).parent / "slams_w.csv"
    if not slams_m_csv.exists() or not slams_w_csv.exists():
        print("  skipping champion_ratings.json (slams_m.csv / slams_w.csv not found)")
        return

    # Build (year, code, tour) -> slam date lookup from matches
    slam_dates = {}
    for code, name in SLAM_NAME.items():
        names = [name] + (["Us Open"] if code == "US" else [])
        for tour in ["ATP", "WTA"]:
            rows = matches[(matches["tourney_name"].isin(names)) & (matches["tour"] == tour)]
            for d in rows["date"].unique():
                slam_dates[(pd.Timestamp(d).year, code, tour)] = pd.Timestamp(d)

    out = {}
    for tour_letter, tour_full, csv_path in [("M", "ATP", slams_m_csv), ("W", "WTA", slams_w_csv)]:
        slams = pd.read_csv(csv_path)
        for _, row in slams.iterrows():
            yr = int(row["year"])
            raw_code = row["slam"]
            winner = str(row["winner"])
            code = CSV_CODE_MAP.get(raw_code, raw_code)
            snap = slam_dates.get((yr, code, tour_full))
            if snap is None:
                continue
            match = ratings[
                (ratings["tour"] == tour_full)
                & (ratings["snapshot_date"] == snap)
                & (ratings["player"] == winner)
            ]
            if match.empty:
                # Loose fallback: last-name contains (handles unicode/diacritic variants)
                last = winner.split()[-1]
                match = ratings[
                    (ratings["tour"] == tour_full)
                    & (ratings["snapshot_date"] == snap)
                    & (ratings["player"].str.contains(last, regex=False, na=False))
                ]
            if match.empty:
                continue
            r = match.iloc[0]
            surface = SLAM_SURFACE[code]
            delta = float(r[f"{surface}_delta"])
            base = float(r["base"])
            out[f"{tour_letter}_{yr}_{code}"] = {
                "player":        r["player"],
                "base":          round(base, 3),
                "surface_delta": round(delta, 3),
                "surface":       surface,
                "effective":     round(base + delta, 3),
            }
    with open(DOCS_DATA / "champion_ratings.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  wrote champion_ratings.json ({len(out)} entries)")


def _slug(name: str) -> str:
    return re.sub(r"[^\w]", "_", name).strip("_")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    args = set(sys.argv[1:])
    # --full forces a complete re-solve (every (tour, snapshot_date) pair).
    # Without it, generate runs INCREMENTALLY: snapshots older than
    # INCREMENTAL_STALENESS_DAYS (30) that already exist in the cached
    # tennis_ratings.csv.gz are reused, and only the recent tail + brand-new
    # anchors get re-solved. Used by the daily cron; the annual Dec 31 run
    # passes --full to re-canvass everything against retroactive Sackmann
    # corrections.
    full_rebuild = "--full" in args
    args.discard("--full")
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
        print(f"\n=== Generating JSON snapshots for SPA "
              f"({'full rebuild' if full_rebuild else 'incremental'}) ===")
        generate_data(full_rebuild=full_rebuild)
