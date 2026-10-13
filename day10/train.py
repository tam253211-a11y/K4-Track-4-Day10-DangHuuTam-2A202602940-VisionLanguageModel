"""Vòng huấn luyện E1/E2/E3/E4 (GUIDE bước 6).

Ví dụ:
    python -m day10.train --config configs/e1.yaml --out work/e1
    python -m day10.train --config configs/e1.yaml --out work/e4_data_0.1 --train-fraction 0.1
    python -m day10.train --config configs/e1.yaml --out work/e4_rank_4 --rank 4
    python -m day10.train --config configs/e1.yaml --out work/smoke --max-steps 3   # chạy thử
"""
import argparse
import math
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from day10.common import MODEL_ID, load_config, load_split, write_json
from day10.data import GroundingDataset, make_collate, num_image_tokens
from day10.modeling import apply_setup, load_base, load_processor, save_projector


def cosine_with_warmup(opt, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def infinite(loader):
    while True:
        yield from loader


def sanity_check(ds, processor):
    """GUIDE bước 4 · Kiểm tra: 220 token ảnh, và phần labels != -100 giải mã ra đúng JSON đáp án."""
    ex = ds[0]
    n_img = num_image_tokens(ex["image_grid_thw"])
    ans = processor.tokenizer.decode(ex["input_ids"][ex["labels"] != -100])
    print(f"[check] image_grid_thw={ex['image_grid_thw'].tolist()} → {n_img} token ảnh; "
          f"seq={ex['input_ids'].numel()} token, học {int((ex['labels'] != -100).sum())} token")
    print(f"[check] labels giải mã: {ans!r}")
    return n_img, ans


def backup(src, backup_dir):
    if backup_dir:
        dst = Path(backup_dir) / Path(src).name
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"[backup] {src} → {dst}")


def save_checkpoint(model, out, sub, train_projector):
    d = Path(out) / sub
    model.save_pretrained(str(d))           # chỉ lưu adapter LoRA
    if train_projector:
        n = save_projector(model, Path(out) / ("projector.pt" if sub == "adapter" else f"{sub}_projector.pt"))
        print(f"[save] {d} + projector ({n} tensor)")
    else:
        print(f"[save] {d}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="tên thí nghiệm (mặc định lấy từ config)")
    ap.add_argument("--rank", type=int, help="E4 rank: ghi đè lora.r, alpha = 2r")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="E4 data_size")
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"], help="E4 coord_scale")
    ap.add_argument("--max-steps", type=int, help="ghi đè max_steps (chỉ để chạy thử)")
    ap.add_argument("--batch-size", type=int, help="ghi đè (khi OOM: 1 kèm --grad-accum 8)")
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--backup-dir", help="vd /content/drive/MyDrive/day10_vlm")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    lora = cfg["lora"]
    r = a.rank or lora["r"]
    alpha = 2 * r if a.rank else lora["alpha"]
    bs = a.batch_size or cfg["batch_size"]
    accum = a.grad_accum or cfg["grad_accum"]
    max_steps = a.max_steps or cfg["max_steps"]
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    records = load_split("train")
    if a.train_fraction < 1.0:
        k = round(len(records) * a.train_fraction)
        records = random.Random(seed).sample(records, k)
    print(f"[data] {len(records)} ảnh train, coord={a.coord}")

    processor = load_processor(cfg.get("model_id", MODEL_ID))
    ds = GroundingDataset(records, processor, coord=a.coord)
    n_img_tokens, _ = sanity_check(ds, processor)

    model = load_base(cfg.get("model_id", MODEL_ID))
    model, info = apply_setup(model, targets=lora["targets"], train_projector=cfg["train_projector"],
                              r=r, alpha=alpha, dropout=lora["dropout"])

    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                        collate_fn=make_collate(processor.tokenizer.pad_token_id),
                        num_workers=a.num_workers, persistent_workers=a.num_workers > 0,
                        generator=torch.Generator().manual_seed(seed))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = cosine_with_warmup(opt, max(1, int(0.05 * max_steps)), max_steps)
    scaler = torch.amp.GradScaler("cuda")
    model.train()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    loss_log, it, t0 = [], infinite(loader), time.time()
    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(accum):
            batch = {k: v.to(model.device) for k, v in next(it).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn ở bước {step} ({loss.item()}): "
                                         "nguy cơ tràn số fp16 — giảm lr / kiểm tra mask")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        loss_log.append(round(step_loss, 5))
        if step % 10 == 0 or step == 1 or step == max_steps:
            el = (time.time() - t0) / 60
            print(f"step {step:4d}/{max_steps} loss={step_loss:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"gnorm={gnorm:.2f} {el:.1f}ph (còn ~{el / step * (max_steps - step):.1f}ph) "
                  f"vram_peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
        if step % a.save_every == 0 and step < max_steps:
            save_checkpoint(model, out, f"ckpt-{step}", cfg["train_projector"])
            backup(out, a.backup_dir)

    train_minutes = (time.time() - t0) / 60
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    save_checkpoint(model, out, "adapter", cfg["train_projector"])

    run = {
        "name": a.name or cfg["name"], "config": str(a.config), "model_id": cfg.get("model_id", MODEL_ID),
        "targets": lora["targets"], "train_projector": cfg["train_projector"],
        "r": r, "alpha": alpha, "dropout": lora["dropout"], "lr": cfg["lr"],
        "batch_size": bs, "grad_accum": accum, "max_steps": max_steps, "seed": seed,
        "coord": a.coord, "train_fraction": a.train_fraction, "n_train_images": len(records),
        "epochs": round(max_steps * bs * accum / len(records), 2), "image_tokens": n_img_tokens,
        **info,
        "peak_vram_gb": round(peak_vram_gb, 2), "train_minutes": round(train_minutes, 1),
        "gpu": torch.cuda.get_device_name(0), "loss_first10": sum(loss_log[:10]) / len(loss_log[:10]),
        "loss_last10": sum(loss_log[-10:]) / len(loss_log[-10:]), "loss_log": loss_log,
    }
    write_json(out / "run.json", run)
    backup(out, a.backup_dir)
    print(f"[done] {run['name']}: trainable={run['trainable_params']:,} peak_vram={run['peak_vram_gb']}GB "
          f"time={run['train_minutes']}ph loss {run['loss_first10']:.3f} → {run['loss_last10']:.3f}")


if __name__ == "__main__":
    main()
