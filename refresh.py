"""Refresh the TENNIS site after a Grand Slam concludes.

Runs both scrapers and re-bundles the timelines data into slams.json
(root + docs/data, the copy fakeronjan-com's native port fetches).
After this finishes successfully, commit and push.

Usage:
    python3 refresh.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(script):
    print(f"\n=== {script} ===")
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nABORT: {script} failed (exit {result.returncode})")


def rebundle():
    print("\n=== bundling timelines into slams.json ===")
    bundle_path = ROOT / "slams.json"
    bundle = json.load(open(bundle_path))
    timelines = json.load(open(ROOT / "player_timelines.json"))
    bundle["timelines"] = timelines
    json.dump(bundle, open(bundle_path, "w"), ensure_ascii=False)
    print(f"  merged {len(timelines)} player timelines into {bundle_path.relative_to(ROOT)}")

    # Also publish under docs/data/ so it's fetchable via GitHub Pages
    # (fakeronjan-com's native port fetches this copy directly).
    data_out = ROOT / "docs" / "data" / "slams.json"
    json.dump(bundle, open(data_out, "w"), ensure_ascii=False)
    print(f"  published {data_out.relative_to(ROOT)}")


def main():
    run("scrape.py")                  # champions list -> slams_m.csv, slams_w.csv, slams.json
    run("build_player_timelines.py")  # per-player timelines from Sackmann match data
                                       # (every 1+ slam winner auto-included; no Wikipedia scrape).
    rebundle()
    print("\nDONE. Next: git add -A && git commit -m 'data: refresh through <slam name> YYYY' && git push")


if __name__ == "__main__":
    main()
