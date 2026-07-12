"""Throwaway: OLMoTrace matched-span length distribution from the disk cache.

Tells us whether the Playground API already returns only long/distinctive spans
(so our MIN_MATCH_CHARS filter is a no-op and we can trust it) or includes short
coincidental matches (so a threshold sweep genuinely matters). Reads the raw
cached responses only — no API calls.

Run from the repo:
    .venv311\\Scripts\\python.exe inspect_olmotrace.py
"""
import json
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "data" / "attribution" / "_olmotrace_cache"


def main() -> None:
    files = sorted(CACHE.glob("*.json"))
    lens: list[int] = []
    samples: list[tuple[int, str]] = []
    n_docs = n_pre = 0
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for doc in d.get("documents", []):
            n_docs += 1
            if doc.get("usage") == "Pre-training":
                n_pre += 1
            for s in doc.get("correspondingSpanTexts", []):
                s = s.strip()
                lens.append(len(s))
                samples.append((len(s), s))

    print(f"cache dir: {CACHE}")
    print(f"{len(files)} cached responses; {n_docs} doc-matches "
          f"({n_pre} pre-training); {len(lens)} matched spans")
    if not lens:
        print("No matched spans cached yet — run build_2x2 (even --limit 6) first.")
        return

    a = np.array(lens)
    for p in (5, 10, 25, 50, 75, 90, 95):
        print(f"  span len p{p:>2} = {np.percentile(a, p):.0f} chars")
    print(f"  share <16 chars = {(a < 16).mean():.2f}   "
          f"<30 = {(a < 30).mean():.2f}   <50 = {(a < 50).mean():.2f}")

    samples.sort(key=lambda t: t[0])
    seen: set[str] = set()
    print("\nshortest distinct matched spans (are these coincidental common phrases?):")
    for n, s in samples:
        if s in seen:
            continue
        seen.add(s)
        print(f"  [{n:>3}] {s!r}")
        if len(seen) >= 6:
            break
    print("\nlongest matched spans (distinctive verbatim training-data evidence?):")
    for n, s in sorted(set(samples), key=lambda t: -t[0])[:3]:
        print(f"  [{n:>3}] {s[:140]!r}")


if __name__ == "__main__":
    main()
