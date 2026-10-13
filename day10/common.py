"""Nhãn, prompt và cấu hình dùng chung (GUIDE bước 2)."""
import json
from pathlib import Path

import yaml

from scripts.labcommon import CLASSES, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

PROMPT_NORM1000 = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)

# E4 coord_scale=pixel: đổi CẢ prompt lẫn nhãn, và chấm bằng --coord pixel
PROMPT_PIXEL = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "in absolute pixels of the {w}x{h} image (x from 0 to {w}, y from 0 to {h}): "
    '[{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}]. '
    "If no damage is found, output []."
)


def get_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        return PROMPT_NORM1000
    if coord == "pixel":
        return PROMPT_PIXEL.format(w=width, h=height)
    raise ValueError(coord)


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (xc, yc, w, h chuẩn hoá [0,1]) → {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)

    def clamp(v, hi):
        return int(round(max(0.0, min(float(hi), v))))

    x1, y1 = clamp((xc - w / 2) * sx, sx), clamp((yc - h / 2) * sy, sy)
    x2, y2 = clamp((xc + w / 2) * sx, sx), clamp((yc + h / 2) * sy, sy)
    # box suy biến (x2<=x1) bị score.parse_output coi là JSON hỏng → nới ra 1 đơn vị
    if x2 <= x1:
        x1, x2 = (x1, x1 + 1) if x1 < sx else (x1 - 1, x1)
    if y2 <= y1:
        y1, y2 = (y1, y1 + 1) if y1 < sy else (y1 - 1, y1)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên→dưới, trái→phải
    return json.dumps(boxes, separators=(",", ":"))


def gt_boxes_norm1000(objects):
    """GT theo đúng định dạng score.py dùng nội bộ: [{"label", "bbox": (x1,y1,x2,y2)}] thang 0–1000."""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": (
            (xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def load_split(name, data_dir=DATA):
    return read_jsonl(Path(data_dir) / "splits" / f"{name}.jsonl")


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["CLASSES", "ROOT", "DATA", "MODEL_ID", "get_prompt", "yolo_to_box", "objects_to_answer",
           "gt_boxes_norm1000", "load_split", "load_config", "read_json", "write_json"]
