"""Chấm bằng scripts/score.py (trọng tài của đề) với --meta lấy NGUYÊN VĂN từ run.json.

Không bao giờ gõ tay số meta: rubric B đối chiếu VRAM/thời gian trong báo cáo với meta.
    python -m day10.evidence --pred work/e0.jsonl --name E0 --out report/evidence/e0.json
    python -m day10.evidence --pred work/e1.jsonl --name E1 --out report/evidence/e1.json --run work/e1
"""
import argparse
import subprocess
import sys
from pathlib import Path

from day10.common import ROOT, read_json

META_KEYS = ("trainable_params", "peak_vram_gb", "train_minutes")


def score(pred, name, out, run=None, split="test", coord=None):
    cmd = [sys.executable, str(ROOT / "scripts" / "score.py"), "--pred", str(pred), "--name", name,
           "--out", str(out), "--split", split]
    if run:
        r = read_json(Path(run) / "run.json")
        coord = coord or r.get("coord", "norm1000")
        for k in META_KEYS:
            v = r[k]
            # score._parse_meta: có dấu "." → float, không có → int
            cmd += ["--meta", f"{k}={int(v)}" if k == "trainable_params" else f"{k}={float(v)}"]
    cmd += ["--coord", coord or "norm1000"]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    return read_json(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--coord", choices=["norm1000", "pixel"])
    a = ap.parse_args(argv)
    score(a.pred, a.name, a.out, a.run, a.split, a.coord)


if __name__ == "__main__":
    main()
