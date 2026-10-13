"""Nhãn YOLO → JSON khớp với trọng tài scripts/score.py (Câu 3). Không cần GPU."""
import json

from day10.common import get_prompt, gt_boxes_norm1000, objects_to_answer, yolo_to_box
from scripts.score import evaluate, parse_output

OBJS = [
    {"label": "crack", "yolo": [0.5, 0.8, 0.3, 0.05]},
    {"label": "pothole", "yolo": [0.25, 0.6, 0.05, 0.04]},
    {"label": "manhole", "yolo": [0.7, 0.6, 0.06, 0.03]},
]


def _entry(objs, iid="a"):
    return {"image_id": iid, "width": 640, "height": 360, "objects": objs}


def test_yolo_to_box_norm1000():
    assert yolo_to_box("pothole", [0.25, 0.6, 0.05, 0.04]) == {"label": "pothole", "bbox_2d": [225, 580, 275, 620]}


def test_yolo_to_box_pixel():
    assert yolo_to_box("pothole", [0.25, 0.6, 0.05, 0.04], "pixel", 640, 360)["bbox_2d"] == [144, 209, 176, 223]


def test_clamped_and_never_degenerate():
    b = yolo_to_box("crack", [0.999, 0.5, 0.0001, 0.0001])["bbox_2d"]
    assert 0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000
    b = yolo_to_box("crack", [1.0, 1.0, 0.1, 0.1])["bbox_2d"]
    assert b[2] == 1000 and b[3] == 1000 and b[0] < b[2]
    b = yolo_to_box("crack", [1.0, 0.5, 0.0, 0.0])["bbox_2d"]          # sát mép phải, w=0
    assert b[0] < b[2] <= 1000


def test_answer_sorted_top_to_bottom_left_to_right_and_compact():
    s = objects_to_answer(OBJS)
    assert " " not in s
    boxes = json.loads(s)
    assert [b["label"] for b in boxes] == ["pothole", "manhole", "crack"]
    assert objects_to_answer([]) == "[]"


def test_ground_truth_answer_scores_perfect_norm1000():
    res = evaluate([_entry(OBJS)], {"a": objects_to_answer(OBJS)}, coord="norm1000", n_boot=10)
    assert res["parse_valid_rate"] == 1.0 and res["micro"]["f1"] == 1.0


def test_ground_truth_answer_scores_perfect_pixel():
    ans = objects_to_answer(OBJS, "pixel", 640, 360)
    assert evaluate([_entry(OBJS)], {"a": ans}, coord="pixel", n_boot=10)["micro"]["f1"] == 1.0


def test_scale_mismatch_kills_f1():
    """Câu 3: nhãn pixel nhưng chấm như 0–1000 (và ngược lại) → F1 sụp dù JSON hợp lệ."""
    pix = objects_to_answer(OBJS, "pixel", 640, 360)
    res = evaluate([_entry(OBJS)], {"a": pix}, coord="norm1000", n_boot=10)
    assert res["parse_valid_rate"] == 1.0 and res["micro"]["f1"] == 0.0
    norm = objects_to_answer(OBJS)
    assert evaluate([_entry(OBJS)], {"a": norm}, coord="pixel", n_boot=10)["micro"]["f1"] == 0.0


def test_gt_boxes_match_score_internal_format():
    parsed = parse_output(objects_to_answer(OBJS))
    assert {p["label"] for p in parsed} == {g["label"] for g in gt_boxes_norm1000(OBJS)}


def test_prompts():
    assert "0-1000" in get_prompt("norm1000")
    p = get_prompt("pixel", 640, 360)
    assert "640x360" in p and '"bbox_2d"' in p
