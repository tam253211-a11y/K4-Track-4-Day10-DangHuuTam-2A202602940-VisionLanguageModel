#!/usr/bin/env bash
# Chạy toàn bộ lab trên máy GPU Linux (Vast.ai / Colab terminal). Chạy lại được: bước nào xong thì bỏ qua.
#
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...        # hoặc: export DATA_ZIP=/path/dataset.zip
#   SMOKE=1 bash run_all.sh      # ~10 phút: kiểm tra pipeline với 3 bước train + 8 ảnh val (KHÔNG ghi report/)
#   bash run_all.sh              # chạy thật: E0 → E1 → E2 → E3 → E4 (data_size) → bảng/ảnh
#   E5=1 bash run_all.sh         # thêm GRPO (tuỳ chọn, +10)
#   BACKUP_DIR=/content/drive/MyDrive/day10_vlm bash run_all.sh   # sao lưu adapter
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
mkdir -p work/logs report/evidence report/figures
BK=${BACKUP_DIR:+--backup-dir $BACKUP_DIR}
log() { echo -e "\n========== $* ==========" | tee -a work/logs/run_all.log; }

# ---------- 0. môi trường
if [ "${SKIP_INSTALL:-0}" != 1 ]; then
  log "cài thư viện"
  pip -q install -r requirements-colab.txt pytest matplotlib datasets requests
fi
nvidia-smi -L
python -m pytest -q tests day10/tests

# ---------- 1. dữ liệu
if [ ! -f data/splits/meta.json ]; then
  log "chuẩn bị dữ liệu"
  if [ -n "${DATA_ZIP:-}" ]; then python scripts/prepare_data.py --zip "$DATA_ZIP"
  else python scripts/prepare_data.py --kaggle; fi
fi
python -m day10.check_data

# ---------- SMOKE: kiểm tra toàn bộ đường đi với chi phí tối thiểu
if [ "${SMOKE:-0}" = 1 ]; then
  log "SMOKE: E0 8 ảnh val"
  python -m day10.infer --zero-shot --split val --limit 8 --out work/smoke/e0_val8.jsonl
  python -m day10.evidence --pred work/smoke/e0_val8.jsonl --name E0smoke --split val --out work/smoke/e0_val8.json
  for cfg in e1 e2 e3; do
    log "SMOKE: $cfg 3 bước"
    python -m day10.train --config configs/$cfg.yaml --out work/smoke/$cfg --max-steps 3 --save-every 2 \
      2>&1 | tee work/logs/smoke_$cfg.log
    python -m day10.infer --run work/smoke/$cfg --split val --limit 8 --out work/smoke/${cfg}_val8.jsonl
    python -m day10.evidence --pred work/smoke/${cfg}_val8.jsonl --name ${cfg^^}smoke --split val \
      --run work/smoke/$cfg --out work/smoke/${cfg}_val8.json
  done
  if [ "${E5:-0}" = 1 ]; then
    log "SMOKE: GRPO 2 bước"
    python -m day10.grpo --init work/smoke/e1 --out work/smoke/e5 --max-steps 2 --n-images 16
  fi
  log "SMOKE xong — xem số token ảnh (phải 220), labels giải mã, số module LoRA (252/252/348), VRAM ở trên"
  exit 0
fi

# ---------- 2–3. E0 zero-shot
if [ ! -f report/evidence/e0.json ]; then
  log "E0 zero-shot"
  python -m day10.infer --zero-shot --out work/e0.jsonl 2>&1 | tee work/logs/e0_infer.log
  python -m day10.evidence --pred work/e0.jsonl --name E0 --out report/evidence/e0.json
fi

# ---------- 4–7. E1, E2, E3
for cfg in e1 e2 e3; do
  N=${cfg^^}
  if [ ! -f report/evidence/$cfg.json ]; then
    if [ ! -f work/$cfg/run.json ]; then
      log "$N train"
      python -m day10.train --config configs/$cfg.yaml --out work/$cfg $BK 2>&1 | tee work/logs/${cfg}_train.log
    fi
    log "$N infer + score"
    python -m day10.infer --run work/$cfg --out work/$cfg.jsonl 2>&1 | tee work/logs/${cfg}_infer.log
    python -m day10.evidence --pred work/$cfg.jsonl --name $N --run work/$cfg --out report/evidence/$cfg.json
  fi
done

# ---------- 8. E4 ablation data_size trên setup tốt nhất E1–E3
BEST=$(python - <<'EOF'
import json
f = {k: json.load(open(f"report/evidence/{k}.json"))["micro"]["f1"] for k in ("e1", "e2", "e3")}
print(max(f, key=f.get))
EOF
)
BEST=${E4_BASE:-$BEST}
log "E4 data_size trên $BEST (đặt E4_BASE=e1 để ép setup gốc)"
if [ ! -f report/evidence/e4_data_1.0.json ]; then   # 100% = chính setup gốc, chỉ chấm lại với tên E4
  python -m day10.evidence --pred work/$BEST.jsonl --name E4_data_1.0 --run work/$BEST \
    --out report/evidence/e4_data_1.0.json
fi
if [ ! -f report/evidence/e4_data_0.1.json ]; then
  if [ ! -f work/e4_data_0.1/run.json ]; then
    python -m day10.train --config configs/$BEST.yaml --out work/e4_data_0.1 --name E4_data_0.1 \
      --train-fraction 0.1 $BK 2>&1 | tee work/logs/e4_data_0.1_train.log
  fi
  python -m day10.infer --run work/e4_data_0.1 --out work/e4_data_0.1.jsonl
  python -m day10.evidence --pred work/e4_data_0.1.jsonl --name E4_data_0.1 --run work/e4_data_0.1 \
    --out report/evidence/e4_data_0.1.json
fi

# ---------- 9. E5 GRPO (tuỳ chọn)
if [ "${E5:-0}" = 1 ] && [ ! -f report/evidence/e5.json ]; then
  log "E5 GRPO từ $BEST"
  python -m pytest -q day10/tests/test_reward.py
  [ -f work/e5/run.json ] || python -m day10.grpo --init work/$BEST --out work/e5 $BK 2>&1 | tee work/logs/e5_grpo.log
  python -m day10.infer --run work/e5 --out work/e5.jsonl
  python -m day10.evidence --pred work/e5.jsonl --name E5 --run work/e5 --out report/evidence/e5.json
fi

# ---------- phân tích cho REPORT.md
log "phân tích"
{
  python -m day10.analyze table
  python -m day10.analyze e0 --pred work/e0.jsonl
  python -m day10.analyze classes --pred work/$BEST.jsonl
  python -m day10.analyze figures --pred work/$BEST.jsonl --tag $BEST
  python -m day10.analyze figures --pred work/e0.jsonl --tag e0
  [ -f work/e5.jsonl ] && python -m day10.analyze hacking --pred work/$BEST.jsonl work/e5.jsonl || true
  python -m day10.coord_check --pred work/$BEST.jsonl
} 2>&1 | tee report/analysis.txt
for f in work/e*/run.json; do mkdir -p report/runs; cp "$f" "report/runs/$(basename "$(dirname "$f")").json"; done
cp work/*.info.json report/runs/ 2>/dev/null || true
log "XONG. Tải về: report/ (evidence, figures, runs, analysis.txt) và work/logs/ — KHÔNG tải work/ vào git"
