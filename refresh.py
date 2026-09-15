"""Refresh the tennis site after a Grand Slam concludes.

Re-scrapes the Wikipedia champions lists (slams_m.csv, slams_w.csv, slams.json)
and publishes the copy under docs/data/ that fakeronjan-com's native port
fetches. Run this by hand 4x/year, after each major's final.

Usage:
    python3 refresh.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    result = subprocess.run([sys.executable, str(ROOT / "scrape.py")], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nABORT: scrape.py failed (exit {result.returncode})")

    shutil.copyfile(ROOT / "slams.json", ROOT / "docs" / "data" / "slams.json")
    print("\npublished docs/data/slams.json")
    print("\nDONE. Next: git add -A && git commit -m 'data: refresh through <slam name> YYYY' && git push")


if __name__ == "__main__":
    main()
