# Phase4 Vision Rescue Runbook (Run D)

## 1) Preflight checks

```bash
cd /home/nate/GemmPali

# environment sanity
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
python -c "import peft; print(peft.__version__)"
python -c "import pyarrow; print(pyarrow.__version__)"

# GPUs (expect 2x3090 lane free)
nvidia-smi

# static syntax checks
python -m py_compile scripts/train_phase4_vision.py
python -m py_compile scripts/run_retrieval_harness_raw_image.py
```

If `peft` import fails, install in the active env before launch:

```bash
pip install peft
```

---

## 2) 250-step smoke command (vision path)

```bash
cd /home/nate/GemmPali
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 scripts/train_phase4_vision.py \
  --pairs nvme_cache/processed/phase3_3_runB/phase3_3_pairs.jsonl \
  --model google/gemma-3-4b-it \
  --steps 250 \
  --batch-size 1 \
  --negatives-per-query 1 \
  --dtype bf16 \
  --max-len 4096 \
  --lr 4e-6 \
  --save-dir checkpoints/phase4_runD_vision_smoke \
  --archive-dir /mnt/ripped_media/GemmPali/checkpoint_archive/phase4_runD_vision_smoke \
  --eval-every 125 \
  --eval-batches 4 \
  --debug-vision-every 25
```

---

## 3) Signals proving vision path is active

During smoke/full runs, confirm all of the following in logs:

1. **`pixel_values_shape` is populated** (from `vision_debug` lines), e.g.:
   - `pixel_values_shape=(B, C, H, W)` or processor-specific packed shape
2. **VRAM jump vs text-only baseline**
   - monitor `nvidia-smi`; vision run should show materially higher usage than phase3 text-surrogate path
3. **Nontrivial late-interaction tensor shapes**
   - `q_seq=(B, Q, D)` and `pos_seq=(B, P, D)` where both Q and P > 1
   - hard negatives show `hard_seq=(B, N, P, D)`
4. **No surrogate-doc assembly in path**
   - trainer uses PIL + AutoProcessor document images; no `query + answer` pseudo-doc concatenation in scoring path

---

## 4) Full-run command

```bash
cd /home/nate/GemmPali
bash scripts/launch_phase4_runD_vision.sh
```

Escalation (only after stable run):
- first raise `--negatives-per-query 1 -> 3`
- then raise `--batch-size 1 -> 2`
- if OOM/regression: immediately revert to `batch-size=1`, `negatives-per-query=1`

---

## 5) Fail-safe rollback steps

1. Stop run cleanly (`Ctrl+C` in foreground launcher or kill torchrun PID).
2. Revert promotion target to last known stable phase3 checkpoint.
3. Keep phase4 artifacts isolated:
   - checkpoints under `checkpoints/phase4_runD_vision*`
   - archives under `/mnt/ripped_media/GemmPali/checkpoint_archive/phase4_runD_vision*`
4. Re-run phase3 harness baseline for sanity comparison.
5. If needed, restart with smoke profile (250 steps) before any long run.
