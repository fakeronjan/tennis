"""Build player_timelines.json from Sackmann match data.

Replaces the older Wikipedia-scraped scrape_players.py - this approach
is faster, never rate-limits, has no name-mismatch issues across years,
and works for every 1+ slam winner in slams.json automatically.

For each eligible player (1+ Open Era slam title), compute the player's
best-finish-by-year at each Grand Slam from the deepest round they
appear in (Sackmann's match data has every main-draw match).

Output: player_timelines.json (consumed by refresh.py's rebundle step).
Each entry: { "Player Name": { "AO": {"YYYY": "<finish>"}, "FO": {...}, "Wim": {...}, "US": {...} } }
Finish codes: W, F, SF, QF, 4R, 3R, 2R, 1R, A (absent from main draw), NH (event not held).

Qualifying rounds (Q1/Q2/Q3) are not surfaced; players absent from
main draw show as "A" regardless of whether they played qualifying.
"""

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

# Wikipedia uses full birth-and-married names for some players; Sackmann uses
# the player's tour name. Manual map covers the edge cases - diacritic
# differences are handled by normalisation.
NAME_MAP = {
    "Ann Haydon Jones":         "Ann Jones",
    "Billie Jean Moffitt King": "Billie Jean King",
    "Evonne Goolagong Cawley":  "Evonne Goolagong",
    "Kerry Melville Reid":      "Kerry Reid",
    "Li Na":                    "Na Li",
    "Margaret Smith Court":     "Margaret Court",
    # Chris O'Neil (1978 AO winner) is absent from Sackmann's data - no timeline.
}

SLAM_NAME_TO_CODE = {
    "Australian Open": "AO",
    "Roland Garros":   "FO",
    "Wimbledon":       "Wim",
    "US Open":         "US",
    "Us Open":         "US",
}

# Sackmann round code -> player-summary timeline code.
ROUND_RANK = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}
ROUND_TO_CODE = {"R128": "1R", "R64": "2R", "R32": "3R", "R16": "4R",
                  "QF": "QF",   "SF": "SF",  "F": "F"}


def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def main():
    matches_path = ROOT / "all_matches.csv"
    slams_path   = ROOT / "slams.json"
    if not matches_path.exists():
        raise SystemExit(f"missing {matches_path} - run tennis_engine.py build first")
    if not slams_path.exists():
        raise SystemExit(f"missing {slams_path} - run scrape.py first")

    m = pd.read_csv(matches_path)
    m["date"] = pd.to_datetime(m["date"])
    m["year"] = m["date"].dt.year
    m["slam_code"] = m["tourney_name"].map(SLAM_NAME_TO_CODE)
    slam_matches = m[m["slam_code"].notna()].copy()

    bundle = json.load(open(slams_path))
    elig = {}
    for tour in ("m", "w"):
        c = Counter(r["w"] for r in bundle["data"][tour])
        elig[tour] = set(n for n, k in c.items() if k >= 1)

    all_sackmann_names = set(m["winner_name"].dropna().unique()) \
                         | set(m["loser_name"].dropna().unique())
    sackmann_normalized = {strip_diacritics(n).lower(): n for n in all_sackmann_names}

    def find_sackmann(name):
        if name in NAME_MAP:
            return NAME_MAP[name]
        if name in all_sackmann_names:
            return name
        return sackmann_normalized.get(strip_diacritics(name).lower())

    slam_year_held = defaultdict(set)
    for (slam_code, tour), grp in slam_matches.groupby(["slam_code", "tour"]):
        for y in grp["year"].unique():
            slam_year_held[(slam_code, tour)].add(int(y))

    def best_finish(sn, slam_code, year, tour):
        sub = slam_matches[(slam_matches["year"] == year)
                            & (slam_matches["slam_code"] == slam_code)
                            & (slam_matches["tour"] == tour)]
        if sub.empty:
            return "NH"
        pm = sub[(sub["winner_name"] == sn) | (sub["loser_name"] == sn)]
        if pm.empty:
            return "A"
        deepest = max(ROUND_RANK.get(r, 0) for r in pm["round"].dropna().tolist())
        deepest_round = next((r for r, n in ROUND_RANK.items() if n == deepest), None)
        if not deepest_round:
            return "A"
        if deepest_round == "F":
            won_final = (pm[pm["round"] == "F"]["winner_name"] == sn).any()
            return "W" if won_final else "F"
        return ROUND_TO_CODE[deepest_round]

    timelines = {}
    missing = []
    for tour_letter in ("m", "w"):
        tour = "ATP" if tour_letter == "m" else "WTA"
        for player in sorted(elig[tour_letter]):
            sn = find_sackmann(player)
            if not sn:
                missing.append(player); continue
            pm = m[(m["tour"] == tour)
                   & ((m["winner_name"] == sn) | (m["loser_name"] == sn))]
            if pm.empty:
                missing.append(player); continue
            yr_min, yr_max = int(pm["year"].min()), int(pm["year"].max())
            tl = {}
            for slam_code in ("AO", "FO", "Wim", "US"):
                slam_tl = {}
                for yr in range(yr_min, yr_max + 1):
                    if yr not in slam_year_held.get((slam_code, tour), set()):
                        slam_tl[str(yr)] = "NH"
                    else:
                        slam_tl[str(yr)] = best_finish(sn, slam_code, yr, tour)
                tl[slam_code] = slam_tl
            timelines[player] = tl

    out_path = ROOT / "player_timelines.json"
    json.dump(timelines, open(out_path, "w"), ensure_ascii=False)
    print(f"Wrote {len(timelines)} player timelines to {out_path.name}")
    if missing:
        print(f"Missing (no Sackmann match data): {missing}")


if __name__ == "__main__":
    main()
