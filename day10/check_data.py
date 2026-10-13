"""Chặng 1 · Kiểm tra split: dataset_version, số ảnh/box khớp đề, không session nào nằm ở hai split.

    python -m day10.check_data            # thêm --figure để vẽ 6 ảnh train có box
"""
import argparse
from collections import Counter
from pathlib import Path

from day10.common import CLASSES, DATA, ROOT, load_split, read_json

EXPECTED = {"train": (1422, 1074, 1750, 691), "val": (276, 66, 470, 103), "test": (311, 121, 299, 163)}
COLORS = {"pothole": (255, 40, 40), "crack": (255, 215, 0), "manhole": (0, 200, 255)}


def check(data_dir=DATA):
    meta = read_json(Path(data_dir) / "splits" / "meta.json")
    print("dataset_version:", meta["dataset_version"])
    assert meta["dataset_version"].startswith("d10-"), "dataset_version phải bắt đầu bằng d10-"
    splits = {s: load_split(s, data_dir) for s in EXPECTED}
    ok = True
    for s, rows in splits.items():
        c = Counter(o["label"] for r in rows for o in r["objects"])
        got = (len(rows), *(c[k] for k in CLASSES))
        empty = sum(1 for r in rows if not r["objects"])
        match = got == EXPECTED[s]
        ok &= match
        print(f"{s:5s} ảnh={got[0]:5d} pothole={got[1]:5d} crack={got[2]:5d} manhole={got[3]:5d} "
              f"ảnh_rỗng={empty:4d} sessions={len({r['session'] for r in rows}):3d}  {'OK' if match else 'KHÁC ĐỀ!'}")
    sess = {s: {r["session"] for r in rows} for s, rows in splits.items()}
    overlap = (sess["train"] & sess["val"]) | (sess["train"] & sess["test"]) | (sess["val"] & sess["test"])
    print("session trùng giữa các split:", sorted(overlap) or "không có")
    assert not overlap, "rò rỉ: có session nằm ở hai split"
    assert ok, "số ảnh/box khác bảng của đề"
    return splits, meta


def draw_gt(rec, data_dir=DATA):
    from PIL import Image, ImageDraw
    img = Image.open(Path(data_dir) / rec["image_path"]).convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = img.size
    for o in rec["objects"]:
        xc, yc, w, h = o["yolo"]
        box = [(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]
        d.rectangle(box, outline=COLORS[o["label"]], width=3)
        d.text((box[0] + 2, max(0, box[1] - 11)), o["label"], fill=COLORS[o["label"]])
    return img


def figure(splits, out=ROOT / "report" / "figures" / "data_samples.png"):
    import matplotlib
    import matplotlib.pyplot as plt
    if "inline" not in matplotlib.get_backend():
        matplotlib.use("Agg")
    samples = [r for r in splits["train"] if r["objects"]][::200][:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    for ax, rec in zip(axes.flat, samples):
        ax.imshow(draw_gt(rec))
        ax.set_title(f"{rec['session']} · {len(rec['objects'])} box", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=80, bbox_inches="tight")
    print("[figure]", out)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args(argv)
    splits, _ = check()
    if a.figure:
        figure(splits)


if __name__ == "__main__":
    main()
