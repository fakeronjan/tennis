"""Refresh the TENNIS site after a Grand Slam concludes.

Runs both scrapers and re-embeds the bundled data into docs/index.html.
After this finishes successfully, commit and push.

Usage:
    python3 refresh.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
HTML = ROOT / "docs" / "index.html"


def run(script):
    print(f"\n=== {script} ===")
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nABORT: {script} failed (exit {result.returncode})")


def rebundle():
    print("\n=== bundling into docs/index.html ===")
    bundle = json.load(open(ROOT / "slams.json"))
    timelines = json.load(open(ROOT / "player_timelines.json"))
    bundle["timelines"] = timelines
    json.dump(bundle, open(ROOT / "slams.json", "w"), ensure_ascii=False)

    html = HTML.read_text()
    new_line = "const BUNDLE = " + json.dumps(bundle, ensure_ascii=False) + ";"
    html, n = re.subn(r"const BUNDLE = \{.*?\};\n", new_line + "\n", html, count=1, flags=re.DOTALL)
    if n != 1:
        sys.exit("ABORT: could not locate BUNDLE line in docs/index.html")
    HTML.write_text(html)
    print(f"  embedded {len(timelines)} player timelines into {HTML.relative_to(ROOT)}")


def main():
    run("scrape.py")                  # champions list -> slams_m.csv, slams_w.csv, slams.json
    run("build_player_timelines.py")  # per-player timelines from Sackmann match data
                                       # (every 1+ slam winner auto-included; no Wikipedia scrape).
    rebundle()
    print("\nDONE. Next: git add -A && git commit -m 'data: refresh through <slam name> YYYY' && git push")


if __name__ == "__main__":
    main()
