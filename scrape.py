"""Scrape Open Era Grand Slam singles champions (men + women) from Wikipedia.

Writes slams_m.csv and slams_w.csv with columns: year, slam, winner, slam_num.
Slam order per year: AO, RG (French), Wim, US.
Filters to 1968+ (Open Era).
"""

import csv
import re
import urllib.request
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
    """Return (name, country_wiki_slug) for a champion-cell, or (None, None) for not-held."""
    if "table-na" in (td.get("class") or []):
        return None, None
    # Country from flagicon link (e.g. /wiki/Australia)
    country = None
    flag = td.select_one(".flagicon a")
    if flag and flag.get("href", "").startswith("/wiki/"):
        country = flag["href"][len("/wiki/"):]
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
    # Each grid cell: (td, text, origin_row) — origin_row is where rowspan starts.
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
            # Skip cells inherited via rowspan from an earlier row — that slam has
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


if __name__ == "__main__":
    main()
