"""Suy luận theo batch → work/<name>.jsonl, mỗi dòng {"image_id", "raw"} (GUIDE bước 3 và 7).

Ví dụ:
    python -m day10.infer --zero-shot --out work/e0.jsonl                    # E0
    python -m day10.infer --run work/e1 --out work/e1.jsonl                   # E1–E4
    python -m day10.infer --run work/e1 --split val --limit 40 --out work/e1_val40.jsonl   # gỡ lỗi
Ghi nối (append) và bỏ qua image_id đã có → Colab bị ngắt thì chạy lại lệnh là chạy tiếp.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from day10.common import DATA, get_prompt, load_split, write_json
from day10.data import user_message


@torch.inference_mode()
def generate_batch(model, processor, images, prompts, max_new_tokens=256):
    messages = [user_message(img, p) for img, p in zip(images, prompts)]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # padding bên trái → mọi dòng có cùng độ dài prompt, cắt bỏ phần prompt
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_inference(model, processor, records, out_path, coord="norm1000", batch_size=4,
                  max_new_tokens=256, data_dir=DATA):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["image_id"] for line in f if line.strip()}
    todo = [r for r in records if r["image_id"] not in done]
    print(f"[infer] {out_path}: đã có {len(done)}, còn {len(todo)} ảnh")
    model.eval()
    model.config.use_cache = True
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            images = [Image.open(data_dir / r["image_path"]).convert("RGB") for r in batch]
            prompts = [get_prompt(coord, r["width"], r["height"]) for r in batch]
            raws = generate_batch(model, processor, images, prompts, max_new_tokens)
            for r, raw in zip(batch, raws):
                f.write(json.dumps({"image_id": r["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            n = i + len(batch)
            if n % (batch_size * 10) == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / 60:.1f}ph (còn ~{el / n * (len(todo) - n) / 60:.1f}ph)",
                      flush=True)
    return (time.time() - t0) / 60


def show_raw(path, k=5):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows[:k]:
        print(f"--- {r['image_id']}\n{r['raw'][:400]}")


def main(argv=None):
    from day10.modeling import load_base, load_processor, load_trained

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="thư mục run (có run.json, adapter/, projector.pt)")
    g.add_argument("--zero-shot", action="store_true", help="E0: model gốc, không adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args(argv)

    records = load_split(a.split)
    if a.limit:
        records = records[:a.limit]
    if Path(a.out).exists():
        with open(a.out, encoding="utf-8") as f:
            done = {json.loads(line)["image_id"] for line in f if line.strip()}
        if all(r["image_id"] in done for r in records):
            print(f"[skip] {a.out} đã đủ {len(records)} ảnh")
            show_raw(a.out)
            return
    processor = load_processor()
    torch.cuda.reset_peak_memory_stats()
    if a.zero_shot:
        model, coord = load_base(), "norm1000"
    else:
        model, run = load_trained(a.run)
        coord = run.get("coord", "norm1000")
    minutes = run_inference(model, processor, records, a.out, coord=coord, batch_size=a.batch_size,
                            max_new_tokens=a.max_new_tokens)
    write_json(Path(a.out).with_suffix(".info.json"), {
        "split": a.split, "n": len(records), "coord": coord, "infer_minutes": round(minutes, 1),
        "peak_vram_gb_infer": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0)})
    show_raw(a.out)


if __name__ == "__main__":
    main()
