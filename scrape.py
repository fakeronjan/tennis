"""Scrape Open Era Grand Slam singles champions (men + women) from Wikipedia.

Writes slams_m.csv and slams_w.csv with columns: year, slam, winner, slam_num.
Slam order per year: AO, RG (French), Wim, US.
Filters to 1968+ (Open Era).
"""

import csv
import json
import re
import urllib.request
from collections import Counter
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (tennis-mock scraper)"}

URLS = {
    "m": "https://en.wikipedia.org/wiki/List_of_Grand_Slam_men%27s_singles_champions",
    "w": "https://en.wikipedia.org/wiki/List_of_Grand_Slam_women%27s_singles_champions",
}

# 1986 had no Australian Open (tournament moved from Dec→Jan; skipped that calendar year).
# The Wikipedia row for 1986 lists AO as "not held". We accept the table's truth.

SLAM_ORDER = ["AO", "RG", "Wim", "US"]


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read().decode("utf-8")


def extract_winner(td):
    """Return (name, country_code) for a champion-cell, or (None, None) for not-held.
    country_code is either a 3-letter IOC code (from the flagicon's Lua-template
    data-mw, e.g. "GBR") or a wiki page-title slug (e.g. "Australia") if that
    metadata isn't present - resolved against IOC_TO_ISO2 / COUNTRY_ISO respectively
    at aggregation time in bundle_to_json()."""
    if "table-na" in (td.get("class") or []):
        return None, None
    country = None
    flagspan = td.select_one(".flagicon")
    # Modern Wikipedia renders the flag via a {{#invoke:flag|icon|XXX}} Lua
    # template - the IOC code is embedded in data-mw as a positional param
    # (not the anchor href, which used to carry a /wiki/<Country> slug but is
    # now an absolute https://en.wikipedia.org/wiki/<Country> URL that a plain
    # /wiki/-prefix check misses). Try this first; it's exact and markup-stable.
    if flagspan and flagspan.get("data-mw"):
        try:
            mw = json.loads(flagspan["data-mw"])
            params = mw["parts"][0]["template"]["params"]
            for key in sorted(params.keys(), key=lambda k: int(k) if k.isdigit() else 99):
                val = (params[key].get("wt") or "").strip()
                if re.fullmatch(r"[A-Za-z]{2,3}", val) and val.lower() != "icon":
                    country = val.upper()
                    break
        except Exception:
            pass
    if country is None:
        flag = td.select_one(".flagicon a")
        href = flag.get("href", "") if flag else ""
        if "/wiki/" in href:
            country = href.split("/wiki/", 1)[1]
    # Now strip markup to extract the player name
    td = BeautifulSoup(str(td), "html.parser")  # clone
    for tag in td.select("sup, .reference, .flagicon"):
        tag.decompose()
    for span in td.find_all("span"):
        style = span.get("style", "")
        if "font-size: 85%" in style or "font-size:85%" in style:
            span.decompose()
    text = td.get_text(" ", strip=True)
    text = re.sub(r"\s*\(\s*\d+\s*/\s*\d+\s*\)\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text or None), country


MARKERS = ("Open Era", "Not held", "Cancelled", "Canceled", "Not Played", "World War")


def _is_marker(text):
    if not text:
        return True
    t = text.strip()
    return any(m.lower() in t.lower() for m in MARKERS)


def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    section = soup.find(id="Champions_by_year")
    candidates = []
    node = section
    while True:
        node = node.find_next(["table", "h2"])
        if node is None or (node.name == "h2" and node is not section):
            break
        if node.name == "table" and "wikitable" in (node.get("class") or []):
            candidates.append(node)
    if not candidates:
        raise RuntimeError("no wikitables under Champions_by_year")
    table = max(candidates, key=lambda t: len(t.find_all("tr")))

    # 5 logical columns: Year, AO, RG, Wim, US.
    # Cells can rowspan or colspan, so we expand into a grid first.
    raw_rows = []
    for tr in table.find_all("tr"):
        if tr.find_all("th", recursive=False):
            continue
        raw_rows.append(tr.find_all(["td"], recursive=False))

    n = len(raw_rows)
    # Each grid cell: (td, text, origin_row) - origin_row is where rowspan starts.
    grid = [[None] * 5 for _ in range(n)]
    for i, tds in enumerate(raw_rows):
        col = 0
        for td in tds:
            while col < 5 and grid[i][col] is not None:
                col += 1
            if col >= 5:
                break
            rs = int(td.get("rowspan", 1))
            cs = int(td.get("colspan", 1))
            text = td.get_text(" ", strip=True)
            for dr in range(rs):
                for dc in range(cs):
                    if i + dr < n and col + dc < 5:
                        grid[i + dr][col + dc] = (td, text, i)
            col += cs

    rows = []
    for i in range(n):
        year_cell = grid[i][0]
        if not year_cell:
            continue
        ymatch = re.match(r"(\d{4})", year_cell[1])
        if not ymatch:
            continue
        year = int(ymatch.group(1))
        if year < 1968:
            continue
        winners = []
        for c in range(1, 5):
            cell = grid[i][c]
            if not cell:
                winners.append((None, None))
                continue
            td, text, origin = cell
            # Skip cells inherited via rowspan from an earlier row - that slam has
            # already been emitted in its origin row.
            if origin != i:
                winners.append((None, None))
                continue
            if "table-na" in (td.get("class") or []) or _is_marker(text):
                winners.append((None, None))
                continue
            winners.append(extract_winner(td))
        rows.append((year, winners))
    return rows


def main():
    for tour, url in URLS.items():
        print(f"fetching {tour}...")
        html = fetch(url)
        rows = parse(html)
        out = f"slams_{tour}.csv"
        slam_num = 0
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "slam", "winner", "country", "slam_num"])
            for year, winners in rows:
                for s, pair in zip(SLAM_ORDER, winners):
                    name, country = pair
                    if name is None:
                        continue
                    slam_num += 1
                    w.writerow([year, s, name, country or "", slam_num])
        print(f"  wrote {out} ({slam_num} slams)")
    bundle_to_json()


# ----------------------------------------------------------------------------
# Bundle step: convert the two raw CSVs into slams.json, applying our
# conventions (RG -> FO, 2020 reorder for COVID schedule, per-player country
# aggregation). slams.json (+ docs/data/slams.json) is what the site fetches.
# ----------------------------------------------------------------------------

# IOC 3-letter code -> ISO2, for the data-mw flag-template path (primary).
IOC_TO_ISO2 = {
    "USA": "US", "ESP": "ES", "SUI": "CH", "SRB": "RS", "GER": "DE", "FRG": "DE",
    "GDR": "DE", "SWE": "SE", "AUS": "AU", "GBR": "GB", "RUS": "RU", "ITA": "IT",
    "ARG": "AR", "BRA": "BR", "ROU": "RO", "AUT": "AT", "CRO": "HR", "RSA": "ZA",
    "FRA": "FR", "ECU": "EC", "NED": "NL", "CZE": "CZ", "TCH": "CZ", "BEL": "BE",
    "JPN": "JP", "POL": "PL", "CHN": "CN", "BLR": "BY", "KAZ": "KZ", "LAT": "LV",
    "DEN": "DK", "CAN": "CA", "YUG": "RS", "URS": "RU", "MEX": "MX", "BUL": "BG",
    "NOR": "NO", "GRE": "GR", "SVK": "SK", "TUN": "TN", "POR": "PT", "SLO": "SI",
    "UK": "GB",
}

# Wiki page-title slug -> ISO2, for the href-based fallback path only.
COUNTRY_ISO = {
    "United_States": "US", "Spain": "ES", "Sweden": "SE", "Serbia": "RS",
    "Switzerland": "CH", "Australia": "AU", "Czechoslovakia": "CZ",
    "Argentina": "AR", "Italy": "IT", "Russia": "RU", "West_Germany": "DE",
    "Germany": "DE", "Brazil": "BR", "United_Kingdom": "GB", "Romania": "RO",
    "Austria": "AT", "Croatia": "HR", "South_Africa": "ZA", "France": "FR",
    "Ecuador": "EC", "Netherlands": "NL", "Czech_Republic": "CZ",
    "Belgium": "BE", "Japan": "JP", "Poland": "PL", "China": "CN",
    "Belarus": "BY", "Kazakhstan": "KZ", "Latvia": "LV", "Denmark": "DK",
    "Canada": "CA",
    "Socialist_Federal_Republic_of_Yugoslavia": "RS",
    "Federal_Republic_of_Yugoslavia": "RS",
    "Soviet_Union": "RU", "Mexico": "MX",
}

# Players who appear flagless on Wikipedia (neutral-athlete status etc.)
MANUAL_COUNTRY = {"Aryna Sabalenka": "BY", "Mirra Andreeva": "RU"}


def bundle_to_json():
    print("\nbundling slams.json...")
    data = {}
    for tour in ("m", "w"):
        rows = []
        with open(f"slams_{tour}.csv") as f:
            reader = csv.reader(f)
            next(reader)  # header
            for r in reader:
                # Strip trailing surface-change marker (§) from names
                name = re.sub(r"\s*§\s*$", "", r[2])
                rows.append({"y": int(r[0]), "s": r[1], "w": name, "c": r[3] or None})
        # RG -> FO
        for r in rows:
            if r["s"] == "RG":
                r["s"] = "FO"
        # 2020 reorder: AO, US, FO (Wim cancelled, FO postponed past US)
        order = {"AO": 0, "US": 1, "FO": 2, "Wim": 3}
        new, pending = [], []
        for r in rows:
            if r["y"] == 2020:
                pending.append(r)
            else:
                if pending and r["y"] > 2020:
                    pending.sort(key=lambda x: order.get(x["s"], 99))
                    new.extend(pending)
                    pending = []
                new.append(r)
        if pending:
            pending.sort(key=lambda x: order.get(x["s"], 99))
            new.extend(pending)
        data[tour] = new

    # Aggregate player -> most-frequent non-blank country -> ISO2
    players = {"m": {}, "w": {}}
    unknown = set()
    for tour in ("m", "w"):
        by_p = {}
        for r in data[tour]:
            if r["c"]:
                by_p.setdefault(r["w"], Counter())[r["c"]] += 1
        for name in {r["w"] for r in data[tour]}:
            if name in MANUAL_COUNTRY:
                players[tour][name] = MANUAL_COUNTRY[name]
            elif name in by_p:
                slug = by_p[name].most_common(1)[0][0]
                iso = IOC_TO_ISO2.get(slug) or COUNTRY_ISO.get(slug)
                if not iso:
                    unknown.add(slug)
                players[tour][name] = iso or "??"
            else:
                players[tour][name] = "??"

    # Strip per-row country (player-level map carries it)
    for tour in ("m", "w"):
        for r in data[tour]:
            r.pop("c", None)

    if unknown:
        print(f"  WARNING: unmapped country slugs: {unknown}")
    print(f"  m: {len(data['m'])} slams, {len(players['m'])} players")
    print(f"  w: {len(data['w'])} slams, {len(players['w'])} players")

    with open("slams.json", "w") as f:
        json.dump({"data": data, "players": players}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
