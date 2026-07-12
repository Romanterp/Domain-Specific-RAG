"""Throwaway: is CTI actually discriminating, and do OLMoTrace pretraining hits
skew toward LOW-CTI (parametric) or HIGH-CTI (both/passage-echo) spans?

Reads reliance_records.jsonl (per-claim CTI) always; if attribution_2x2.jsonl is
present it also crosses CTI against the OLMoTrace pretraining flag. No API calls.

Run:  .venv311\\Scripts\\python.exe calibrate_cti.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "data" / "reliance_records.jsonl"
TWOBYTWO = ROOT / "data" / "attribution_2x2.jsonl"


def load_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def pct(a, ps=(1, 5, 10, 25, 50, 75, 90, 95)):
    a = np.asarray(a, float)
    return "  ".join(f"p{p}={np.percentile(a, p):.3f}" for p in ps)


def main() -> None:
    recs = load_jsonl(RECORDS)
    cti = [c["cti_mean"] for r in recs for c in r.get("claims", [])
           if c.get("cti_mean") is not None]
    print(f"reliance_records: {len(recs)} records, {len(cti)} claims")
    if cti:
        a = np.array(cti)
        print("per-claim CTI distribution:")
        print("  " + pct(a))
        for t in (0.05, 0.10, 0.20, 0.30, 0.50):
            print(f"  share CTI < {t:.2f} = {(a < t).mean():.3f}")
        print(f"  -> a real low tail (mass below ~0.1) means 'parametric' spans exist "
              f"and CTI can discriminate; near-zero mass means everything is context-driven.")

    rows = load_jsonl(TWOBYTWO)
    if not rows:
        print("\n(no attribution_2x2.jsonl yet — run build_2x2 to get the CTI x pretraining cross)")
        return
    hit = [r["cti_mean"] for r in rows
           if r.get("pretraining_hit") is True and r.get("cti_mean") is not None]
    miss = [r["cti_mean"] for r in rows
            if r.get("pretraining_hit") is False and r.get("cti_mean") is not None]
    print(f"\nattribution_2x2: {len(rows)} spans  "
          f"({len(hit)} pretraining-hit, {len(miss)} no-hit)")
    if hit:
        print(f"  CTI | pretraining-HIT : median={np.median(hit):.3f}   {pct(hit, (25,50,75))}")
    if miss:
        print(f"  CTI | NO hit          : median={np.median(miss):.3f}   {pct(miss, (25,50,75))}")
    print("  -> if pretraining-hit spans skew LOWER CTI than no-hit, the parametric "
          "quadrant is real; if they skew the SAME/higher, hits are mostly passage-echo ('both').")


if __name__ == "__main__":
    main()
