"""Scrape per-player singles Grand Slam performance timeline.

For each eligible player (3+ OE slam wins), fetch their Wikipedia
career_statistics page (falling back to the main article) and extract the
singles performance timeline rows for the four majors.

Writes player_timelines.json with shape:
  {
    "Roger Federer": {
      "AO": {"1999":"Q1", "2000":"3R", ...},
      "FO": {...}, "Wim": {...}, "US": {...}
    },
    ...
  }
"""

import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (tennis-mock player timeline scraper)"}

# Canonical Wikipedia article names (drop maiden, fix conventions).
OVERRIDES = {
    "Margaret Smith Court": "Margaret Court",
    "Billie Jean Moffitt King": "Billie Jean King",
    "Ann Haydon Jones": "Ann Jones",
}

# Tournament name -> our slam code. Older player pages use short country
# labels ("Australia", "France", "United States") instead of the full
# tournament names; alias them in.
TOURNEY_MAP = {
    "Australian Open": "AO",
    "Australia": "AO",
    "French Open": "FO",
    "France": "FO",
    "Wimbledon": "Wim",
    "US Open": "US",
    "U.S. Open": "US",
    "United States": "US",
}


def fetch(url):
    try:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=15
        ).read().decode("utf-8")
    except Exception as e:
        print(f"  fetch failed: {e}")
        return None


def candidate_urls(player):
    base = OVERRIDES.get(player, player)
    slug = urllib.parse.quote(base.replace(" ", "_"))
    return [
        f"https://en.wikipedia.org/wiki/{slug}_career_statistics",
        f"https://en.wikipedia.org/wiki/{slug}",
    ]


def _label_to_slam(label):
    """Match row labels even when they have decoration like 'French Open ( details )'."""
    if not label:
        return None
    s = re.sub(r"[\(\[].*?[\)\]]", "", label).strip()
    s = re.sub(r"[\*\†\‡\§]+$", "", s).strip()
    return TOURNEY_MAP.get(s)


_YEAR_RE = re.compile(r"^\s*(\d{4}|['’]?\d{2})\s*$")


def _looks_like_year(s):
    return bool(s and _YEAR_RE.match(s))


def _years_from_header_rows(rows):
    """Find the year list, aligning with the data row's positions.

    Data rows always have the tournament name in cell 0 and per-year values in
    cells 1+. So the year list must be the same length as those value cells.
    Some tables split the header across two rows (country era band on top,
    years on row 1); some have a single header row.
    """
    def parse(header_cells):
        years = []
        for h in header_cells:
            m = re.search(r"(\d{4})", h)
            if m:
                years.append(int(m.group(1)))
            else:
                m2 = re.match(r"['’]?(\d{2})$", h)
                if m2:
                    yy = int(m2.group(1))
                    years.append(1900 + yy if yy >= 30 else 2000 + yy)
                else:
                    years.append(None)
        return years

    def header_after_label(cells):
        # Skip the leading cell if it isn't itself a year (it's a 'Tournament'
        # label, an empty corner, or an era band).
        if cells and not _looks_like_year(cells[0]):
            return cells[1:]
        return cells

    h0_raw = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
    y0 = parse(header_after_label(h0_raw))
    if sum(1 for y in y0 if y) >= 3:
        return y0
    if len(rows) > 1:
        h1_raw = [c.get_text(" ", strip=True) for c in rows[1].find_all(["th", "td"])]
        y1 = parse(header_after_label(h1_raw))
        if sum(1 for y in y1 if y) >= 3:
            return y1
    return None


def find_singles_table(soup):
    """Return (years, rowmap) for the singles GS timeline table, or None."""
    for table in soup.find_all("table", class_="wikitable"):
        rows = table.find_all("tr")
        if len(rows) < 4:
            continue
        # Collect rows whose first cell matches a slam name (with decoration tolerance)
        rowmap = {}
        for tr in rows:
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            slam = _label_to_slam(cells[0].get_text(" ", strip=True))
            if not slam:
                continue
            vals = []
            for c in cells[1:]:
                tc = BeautifulSoup(str(c), "html.parser")
                for sup in tc.select("sup, .reference"):
                    sup.decompose()
                vals.append(tc.get_text(" ", strip=True))
            if slam not in rowmap:
                rowmap[slam] = vals
        if len(rowmap) < 3:
            continue
        years = _years_from_header_rows(rows)
        if not years:
            continue
        return years, rowmap
    return None


# Normalize cell values to compact codes
def normalize_finish(v):
    if not v:
        return None
    s = v.strip()
    if s in ("", "—", "-", "–"):
        return None
    # Common patterns
    s = re.sub(r"\s+", "", s)
    if s.lower() in ("absent", "a", "n/a", "na", "did", "didnotcompete", "nh"):
        return "A"
    # Round shorthand: 1R/2R/3R/4R/SF/QF/F/W/Q1/Q2
    if re.match(r"^(W|F|SF|QF|1R|2R|3R|4R|Q[123]|LQ|Q|DNQ)$", s, re.I):
        return s.upper()
    # Stray trailing punctuation
    s = re.sub(r"[\^\*\†\‡]+$", "", s)
    return s[:6] if s else None


def parse_timeline(player):
    for url in candidate_urls(player):
        html = fetch(url)
        if html is None:
            continue
        soup = BeautifulSoup(html, "html.parser")
        result = find_singles_table(soup)
        if not result:
            continue
        years, rowmap = result
        timeline = {"AO": {}, "FO": {}, "Wim": {}, "US": {}}
        for slam, vals in rowmap.items():
            for i, v in enumerate(vals):
                if i >= len(years):
                    break
                yr = years[i]
                if yr is None:
                    continue
                code = normalize_finish(v)
                if code:
                    timeline[slam][str(yr)] = code
        return timeline, url
    return None, None


# Per-player manual overrides applied after automatic reconciliation. Each
# entry is a list of (slam, from_year, to_year) moves. Used when a player's
# Wikipedia page uses a labeling convention the generic shift can't handle.
MANUAL_TIMELINE_MOVES = {
    # Wikipedia's Rosewall career-stats uses tournament-start year for the
    # early-70s AOs (which straddled Dec/Jan); our slam list uses end year.
    # Apply +1 shift to his early-70s AO entries.
    "Ken Rosewall": [
        ("AO", 1971, 1972),  # Move first to clear the destination
        ("AO", 1970, 1971),
        ("AO", 1969, 1970),
        ("AO", 1968, 1969),
    ],
}


def apply_manual_overrides(timelines):
    moved = 0
    for player, moves in MANUAL_TIMELINE_MOVES.items():
        tl = timelines.get(player)
        if not tl:
            continue
        for slam, from_y, to_y in moves:
            col = tl.get(slam, {})
            src = str(from_y)
            dst = str(to_y)
            if src in col:
                col[dst] = col.pop(src)
                moved += 1
        tl[slam] = col
    if moved:
        print(f"manual overrides: {moved} cell moves applied")


def reconcile_with_champions(timelines, champions_data):
    """Wikipedia career-stats pages are inconsistent about how they label the
    pre-1987 Australian Open: some use calendar year, others bump December
    editions (1978-1985) into the following year. When a player's page uses
    the season-year convention, we shift only the entries that *actually*
    were Dec-held (years 1978-1986 in raw career-stats) by -1 year. Jan-held
    AOs from 1968-1976 and 1987+ stay put. Also drops pre-OE W cells."""
    truth = {}
    for tour in ("m", "w"):
        for r in champions_data["data"][tour]:
            truth.setdefault((r["w"], r["s"]), set()).add(r["y"])

    shifted_cols, dropped_pre_oe = 0, 0
    for player, tl in timelines.items():
        if not tl:
            continue
        for slam in ("AO", "FO", "Wim", "US"):
            col = tl.get(slam, {})
            if not col:
                continue
            true_years = truth.get((player, slam), set())

            def score_with_range_shift(apply_shift):
                """Count W cells that match truth, assuming the year-range
                shift is applied (or not)."""
                hits = 0
                for y_str, v in col.items():
                    if v != "W":
                        continue
                    y = int(y_str)
                    if apply_shift and 1978 <= y <= 1986:
                        y -= 1
                    if y in true_years:
                        hits += 1
                return hits

            if score_with_range_shift(True) > score_with_range_shift(False):
                shifted = {}
                for y_str, v in col.items():
                    y = int(y_str)
                    target = str(y - 1) if 1978 <= y <= 1986 else y_str
                    if int(target) < 1968:
                        continue
                    if target in shifted:
                        # 1977 may collide (Jan stays at 1977 + shifted Dec
                        # lands at 1977). Prefer a W over anything else.
                        if v == "W" and shifted[target] != "W":
                            shifted[target] = v
                    else:
                        shifted[target] = v
                col = shifted
                shifted_cols += 1

            # Per-cell rescue: any leftover unmatched W tries a ±1 nudge.
            for y_str, val in list(col.items()):
                if val != "W":
                    continue
                y = int(y_str)
                if y in true_years:
                    continue
                for shift in (-1, +1):
                    target = y + shift
                    if target in true_years and col.get(str(target)) != "W":
                        col[str(target)] = "W"
                        del col[y_str]
                        break

            # Drop pre-OE W cells regardless of shift
            for y_str in list(col.keys()):
                if col[y_str] == "W" and int(y_str) < 1968:
                    del col[y_str]
                    dropped_pre_oe += 1
            tl[slam] = col
    print(f"reconciliation: shifted {shifted_cols} columns, dropped {dropped_pre_oe} pre-OE W cells")
    return timelines


def main():
    bundle = json.load(open("slams.json"))
    eligible = set()
    for tour in ("m", "w"):
        c = Counter(r["w"] for r in bundle["data"][tour])
        eligible.update(n for n, k in c.items() if k >= 3)

    timelines = {}
    sources = {}
    for i, name in enumerate(sorted(eligible)):
        print(f"[{i+1}/{len(eligible)}] {name}...")
        timeline, src = parse_timeline(name)
        if timeline is None:
            print(f"  ! NO TIMELINE")
            timelines[name] = None
        else:
            counts = {k: len(v) for k, v in timeline.items()}
            print(f"  ok {counts}")
            timelines[name] = timeline
            sources[name] = src
        time.sleep(0.3)  # be polite to wiki

    timelines = reconcile_with_champions(timelines, bundle)
    apply_manual_overrides(timelines)
    with open("player_timelines.json", "w") as f:
        json.dump(timelines, f, ensure_ascii=False)
    print(f"\nwrote player_timelines.json — {sum(1 for v in timelines.values() if v)} OK / {sum(1 for v in timelines.values() if not v)} failed")


if __name__ == "__main__":
    main()
