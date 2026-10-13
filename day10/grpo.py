"""E5 (tuỳ chọn): GRPO với reward IoU, khởi tạo từ run SFT tốt nhất (GUIDE bước 9, trl==1.9.2).

    python -m day10.grpo --init work/e1 --out work/e5
    python -m day10.infer --run work/e5 --out work/e5.jsonl
    python -m day10.evidence --pred work/e5.jsonl --name E5 --out report/evidence/e5.json --run work/e5

Những điểm đã đối chiếu với mã nguồn trl 1.9.2:
- dataset cần cột "prompt" (dạng hội thoại) + cột "image"; trainer tự chèn ảnh vào message.
  Các cột khác (ở đây "gt") được truyền vào reward function qua **kwargs (remove_unused_columns=False mặc định).
- GRPOConfig mặc định bf16=True khi không đặt fp16 → T4 không có bf16, phải đặt fp16=True, bf16=False.
- Với model lượng tử hoá, GRPOTrainer.__init__ ép mọi tham số requires_grad sang bf16 → ta khôi phục fp32.
- beta=0.0 (mặc định) → không nạp model tham chiếu, tiết kiệm VRAM.
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch

from day10.common import gt_boxes_norm1000, load_config, load_split, read_json, write_json, get_prompt, DATA


def build_dataset(records, coord="norm1000"):
    from datasets import Dataset
    from PIL import Image
    rows = []
    for r in records:
        gt = [{"label": b["label"], "bbox": list(b["bbox"])} for b in gt_boxes_norm1000(r["objects"])]
        rows.append({"prompt": [{"role": "user", "content": get_prompt(coord, r["width"], r["height"])}],
                     "image": Image.open(DATA / r["image_path"]).convert("RGB"),
                     "gt": json.dumps(gt), "image_id": r["image_id"]})
    return Dataset.from_list(rows)


def main(argv=None):
    from trl import GRPOConfig, GRPOTrainer

    from day10.modeling import load_processor, load_trained, trainable_params
    from day10.reward import iou_reward

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="run SFT tốt nhất (vd work/e1)")
    ap.add_argument("--out", default="work/e5")
    ap.add_argument("--config", default="configs/e5.yaml")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--n-images", type=int, default=480, help="số ảnh train lấy mẫu cho GRPO")
    ap.add_argument("--per-device-batch", type=int, default=4, help="số completion mỗi micro-batch")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--backup-dir")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if (Path(a.out) / "run.json").exists() and not a.force:
        print(f"[skip] {a.out}/run.json đã có — bỏ qua (dùng --force để chạy lại)")
        return

    g = load_config(a.config)["grpo"]
    max_steps = a.max_steps or g["max_steps"]
    init_run = read_json(Path(a.init) / "run.json")
    if init_run.get("coord", "norm1000") != "norm1000":
        raise ValueError("reward dùng thang 0–1000: hãy khởi tạo từ run norm1000")

    processor = load_processor()
    model, _ = load_trained(a.init, is_trainable=True)
    # projector (nếu run gốc là E2/E3) đã được nạp nhưng để ĐÓNG BĂNG: GRPO chỉ cập nhật LoRA
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    keep = {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}

    random.seed(42)
    records = [r for r in load_split("train")]
    records = random.Random(42).sample(records, min(a.n_images, len(records)))
    ds = build_dataset(records)
    n_empty = sum(1 for r in records if not r["objects"])
    print(f"[grpo] {len(ds)} ảnh train (rỗng: {n_empty}), init={a.init}, trainable={trainable_params(model):,}")

    out = Path(a.out)
    args = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=max_steps, learning_rate=g["learning_rate"],
        num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
        beta=g["beta"], temperature=a.temperature,
        per_device_train_batch_size=a.per_device_batch, gradient_accumulation_steps=a.grad_accum,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False, logging_steps=1, save_strategy="no", report_to="none", seed=42,
        remove_unused_columns=False, warmup_steps=0, lr_scheduler_type="constant",
    )
    trainer = GRPOTrainer(model=model, reward_funcs=iou_reward, args=args, train_dataset=ds,
                          processing_class=processor)
    # GRPOTrainer đã ép tham số huấn luyện của model 4-bit sang bf16 → khôi phục giá trị fp32 gốc
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in keep:
                p.data = keep[n].to(p.device)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    model.save_pretrained(str(out / "adapter"))
    if init_run.get("train_projector"):
        shutil.copy2(Path(a.init) / "projector.pt", out / "projector.pt")
    log = trainer.state.log_history
    run = {**{k: init_run[k] for k in ("model_id", "targets", "train_projector", "r", "alpha", "coord")},
           "name": "E5", "init": str(a.init), "algo": "GRPO", **g, "max_steps": max_steps,
           "temperature": a.temperature, "n_train_images": len(ds), "n_empty_images": n_empty,
           "trainable_params": trainable_params(model),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "train_minutes": round(minutes, 1), "gpu": torch.cuda.get_device_name(0)}
    write_json(out / "run.json", run)
    write_json(Path("report") / "e5_grpo_log.json", log)    # log GRPO (rubric bonus: "chạy GRPO có log")
    rew = [x.get("reward") for x in log if "reward" in x]
    if rew:
        k = max(1, len(rew) // 5)
        print(f"[grpo] reward trung bình {k} bước đầu {sum(rew[:k]) / k:.3f} → {k} bước cuối {sum(rew[-k:]) / k:.3f}")
    if a.backup_dir:
        shutil.copytree(out, Path(a.backup_dir) / out.name, dirs_exist_ok=True)
    print(f"[done] E5 {minutes:.1f} phút, peak VRAM {run['peak_vram_gb']} GB")


if __name__ == "__main__":
    main()
