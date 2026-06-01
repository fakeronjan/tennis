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
# Each set is one observation. Each player gets 4 unknowns: a base rating +
# 3 surface deltas (hard/clay/grass). Carpet matches still feed the base
# rating but DO NOT get their own per-player delta — the modern tour has
# essentially zero carpet, so per-player carpet deltas over-fit on tiny
# samples and produced phantom +1.4 carpet bumps for players who'd played
# 2 carpet matches in a year. By absorbing carpet into base, we keep all
# the 1980s/90s match data without the noise.

SURFACE_LIST = ["Hard", "Clay", "Grass"]  # surfaces that get per-player deltas
PARAMS_PER_PLAYER = 1 + len(SURFACE_LIST)  # base + N deltas


MIN_SETS_PLAYED = 30  # filter for published rankings — drops tiny-sample noise

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
        snapshot to ~0 at the window's far edge — the standard rolling-rating
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

    # Build X sparsely-ish (still dense numpy — for our 1-year window scale it fits)
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
    #   1. Cross-player zero-sum on BASE — sum of all bases = 0 (anchors the
    #      overall rating scale).
    #   2. Per-player zero-sum on MODELED DELTAS — for each player,
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
# STEP 5 — ROLLING SNAPSHOTS + JSON OUTPUTS
# ============================================================
# Snapshot cadence: end-of-November each year (after ATP/WTA Finals usually) +
# current state. ~52 snapshots × 2 tours = ~100 solves, ~5-10 minutes total.

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
    across the player's entire match log — no window filtering."""
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


def build_rolling_snapshots(obs: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Solve ratings at three kinds of anchor dates:

      - slam:    end date of each Grand Slam, with the standard 365-day rolling
                 window + recency decay (the "form heading into the slam" view).
      - eoy:     Dec 31 of each completed calendar year (or the year's last match
                 date if earlier), with a Jan 1 -> Dec 31 calendar-year window
                 and NO recency decay — the "power within this year" view.
                 Skipped for the current calendar year (in-progress).
      - current: the latest match date in the data (per tour). Standard rolling
                 window + recency decay. Only added if it differs from the latest
                 slam snapshot.

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
    print(f"  {len(slam_pairs)} slam + {len(eoy_pairs)} EOY + "
          f"{len(current_pairs - slam_pairs)} current-only snapshots queued")

    all_rows = []
    last_print_year = None

    # Helper to solve one anchor and collect rows
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

    # Solve slam + current snapshots (rolling window, recency on)
    for tour, snap in sorted(slam_pairs | current_pairs):
        snap_type = "slam" if (tour, snap) in slam_pairs else "current"
        _add(tour, snap, snap_type)
        yr = snap.year
        if yr != last_print_year:
            print(f"  ... year {yr}")
            last_print_year = yr

    # Solve EOY snapshots (calendar-year window, no recency)
    for tour, snap, yr in sorted(eoy_pairs):
        window_start = pd.Timestamp(year=yr, month=1, day=1)
        _add(tour, snap, "eoy", window_start=window_start, recency_weighting=False)

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

    print("\nBuilding slam-day snapshots + current...")
    ratings = build_rolling_snapshots(obs, matches)
    print(f"\n{len(ratings):,} rating rows across {ratings['snapshot_date'].nunique()} snapshots")

    # Save full ratings CSV (gzipped — large)
    ratings_out = ratings.copy()
    ratings_out["snapshot_date"] = ratings_out["snapshot_date"].dt.strftime("%Y-%m-%d")
    ratings_out.to_csv(RATINGS_CSV, index=False, compression="gzip")
    print(f"Wrote {RATINGS_CSV}: {len(ratings_out):,} rows")

    # --- Power Rankings history: ALL snapshots (slam + eoy + current) per tour ---
    # Powers the Year + Within-Year picker on the Power Rankings tab.
    write_power_rankings_history(ratings, matches)

    # --- Current rankings (latest snapshot PER TOUR — ATP and WTA can differ) ---
    for tour in ["ATP", "WTA"]:
        tour_ratings = ratings[ratings["tour"] == tour]
        latest_snap = tour_ratings["snapshot_date"].max()
        t = tour_ratings[tour_ratings["snapshot_date"] == latest_snap].sort_values("base", ascending=False).head(50).reset_index(drop=True)
        rows = []
        for i, r in t.iterrows():
            rows.append({
                "rank":         i + 1,
                "player":       r["player"],
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
    # Both views use the EOY (end-of-year) snapshot — calendar-year window,
    # tier weighting only, no recency decay. One full-season-calibrated
    # rating per (player, year).
    #   PEAK = player's max EOY base rating across career. Surfaces the
    #          peak-year body of work (W-L, titles, slams that year).
    #   ERA  = sum of (player's positive EOY base ratings) across career.
    #          Surfaces career totals (slams, titles, year-end #1 finishes).
    #          Negative years clipped at 0 — body of work above tour average.
    eoy_only = ratings[ratings["snapshot_type"] == "eoy"].copy()
    eoy_only["year"] = pd.to_datetime(eoy_only["snapshot_date"]).dt.year

    # Shared per-player match log for window/career stat lookups.
    per_player = build_player_match_index(matches)

    # Filter low-confidence EOY snapshots. A (player, year) qualifies for GOAT
    # consideration if EITHER:
    #   (a) sets_played >= EOY_MIN_SETS (full-season sample), OR
    #   (b) the player won at least one Grand Slam singles title that year.
    # Rule (a) filters phantom small-sample peaks (e.g. Yen Hsun Lu 2005 at
    # 45 sets / 6-8 record, Graf 1997 at 41 sets after her knee injury).
    # Rule (b) is an escape hatch for shortened slam-winning seasons (Serena
    # Williams 2010, 67 sets but 2 slams).
    EOY_MIN_SETS = 80
    slam_year_winners = set()  # (tour, year, player)
    matches_dated = matches.copy()
    matches_dated["date"] = pd.to_datetime(matches_dated["date"])
    matches_dated["year"] = matches_dated["date"].dt.year
    slam_finals = matches_dated[
        (matches_dated["tourney_name"].isin(SLAM_NAMES))
        & (matches_dated["round"] == "F")
    ]
    for _, r in slam_finals.iterrows():
        slam_year_winners.add((r["tour"], int(r["year"]), r["winner_name"]))

    def qualifies(row):
        if row["sets_played"] >= EOY_MIN_SETS:
            return True
        return (row["tour"], int(row["year"]), row["player"]) in slam_year_winners

    before = len(eoy_only)
    eoy_only = eoy_only[eoy_only.apply(qualifies, axis=1)].copy()
    print(f"  EOY filter: {before:,} -> {len(eoy_only):,} player-years (kept >={EOY_MIN_SETS} sets OR slam-winning year)")

    for tour in ["ATP", "WTA"]:
        t = eoy_only[eoy_only["tour"] == tour].copy()
        if t.empty:
            continue

        # Year-end #1 count per player (by our EOY rating, not official rankings).
        no1_per_year = t.loc[t.groupby("year")["base"].idxmax()]
        no1_counts = no1_per_year.groupby("player").size().to_dict()

        # ----- PEAK -----
        peaks = t.loc[t.groupby("player")["base"].idxmax()].copy()
        peaks = peaks.sort_values("base", ascending=False).head(50).reset_index(drop=True)
        peak_rows = []
        for i, r in peaks.iterrows():
            # Stats from the player's peak EOY year (calendar-year window).
            year = int(r["year"])
            yr_start = pd.Timestamp(year=year, month=1, day=1)
            yr_end   = pd.Timestamp(year=year, month=12, day=31)
            wins, losses, titles, slams = window_stats(
                per_player, tour, r["player"], yr_start, yr_end)
            peak_rows.append({
                "rank":          i + 1,
                "player":        r["player"],
                "peak_year":     year,
                "base":          round(float(r["base"]), 3),
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

        # ----- ERA -----
        # Sum of positive EOY ratings across each player's career.
        t["positive"] = t["base"].clip(lower=0)
        era_score = t.groupby("player").agg(
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
            era_rows.append({
                "rank":             i + 1,
                "player":           r["player"],
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

    # --- Champion ratings overlay (powers Champions tab inline rating) ---
    # Keyed by `{tourLetter}_{year}_{spaCode}` -> {player, base, surface_delta, surface, effective}
    # spaCode mapping: AO=Australian Open, FO=Roland Garros, Wim=Wimbledon, US=US Open
    write_champion_ratings(matches, ratings)

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


def write_power_rankings_history(ratings: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Emit docs/data/power_rankings_history_{atp,wta}.json — every snapshot the
    SPA needs for the Year + Within-Year picker on the Power Rankings tab.

    Snapshot keys:
      - "{year}_AO"   — after Australian Open
      - "{year}_FO"   — after Roland Garros
      - "{year}_Wim"  — after Wimbledon
      - "{year}_US"   — after US Open
      - "{year}_EOY"  — Dec 31 calendar-year snapshot (full year, no recency)
      - "Today"       — latest data point (only for current calendar year)

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
    # Only the "covid" category applies here — tennis had no work stoppage or
    # cancelled-season equivalents to the labor / cancelled tags used elsewhere.
    DISRUPTED_SNAPSHOTS = {
        "2020_FO":  {"tag": "covid", "category": "covid",
                     "note": "Roland Garros 2020 rescheduled to Sept-Oct (normally late May/early June) after the Mar-Aug COVID tour shutdown."},
        "2020_US":  {"tag": "covid", "category": "covid",
                     "note": "US Open 2020 played in the NYC bubble with a thin field — Federer, Nadal, and several other top players sat out; Djokovic was defaulted."},
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
            top = group.sort_values("base", ascending=False).head(50).reset_index(drop=True)
            players = []
            for i, r in top.iterrows():
                wins, losses, titles, slams = window_stats(
                    per_player, tour, r["player"], win_start, win_end)
                players.append({
                    "rank":         i + 1,
                    "player":       r["player"],
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


def write_champion_ratings(matches: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Emit docs/data/champion_ratings.json — used by the Champions tab to show
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
