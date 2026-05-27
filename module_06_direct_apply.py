"""
module_06_direct_apply.py
=========================

Centralized (direct) training of LLaVA-1.5-7B with LoRA under QGTP.

QGTP modes  : off | fixed | student
Key features: cosine-warmup LR, gradient accumulation, early stopping.

Usage
-----
    DATASET_KEY=vqarad QGTP_MODE=off python module_06_direct_apply.py
"""

import os, gc, time, json
import torch, numpy as np
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from huggingface_hub import upload_file

from qgtp_lib import (
    setup_hf, load_dataset_split, split_train_val_test,
    FrozenEncoders, load_student_from_hf, QGTPController, LLaVAWithQGTP,
    train_step, evaluate, cosine_warmup_lr,
    OFFLINE, HF_REPO, HF_TOKEN,
)

# =============================================================================
# CONFIG
# =============================================================================
DATASET_KEY = os.environ.get("DATASET_KEY", "vqarad")
QGTP_MODE   = os.environ.get("QGTP_MODE",   "off")
FIXED_RHO   = float(os.environ.get("FIXED_RHO", "0.5"))

N_ROUNDS         = 20
STEPS_PER_ROUND  = 100
GRAD_ACCUM       = 2
BATCH_SIZE       = 16
EVAL_BATCH_SIZE  = 16
MAX_EVAL_SAMPLES = 100
LR               = 1e-4
WARMUP_FRAC      = 0.05
WEIGHT_DECAY     = 0.0
GRAD_CLIP        = 1.0
LR_MIN_RATIO     = 0.05
LORA_R           = 32
LORA_ALPHA       = 64
LORA_DROPOUT     = 0.05
EARLY_STOP_PATIENCE  = 3
EARLY_STOP_MIN_DELTA = 0.005
MAX_DATASET      = 5000
MAX_NEW_TOKENS   = 10
SEED             = 42

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_TAG = f"direct_{DATASET_KEY}_{QGTP_MODE}"
if QGTP_MODE == "fixed":
    RUN_TAG += f"_rho{FIXED_RHO}"

setup_hf()
torch.manual_seed(SEED)
print(f"[config] {RUN_TAG}  rounds={N_ROUNDS}  bs={BATCH_SIZE}  lr={LR}  lora_r={LORA_R}")

# =============================================================================
# DATA + COMPONENTS
# =============================================================================
samples = load_dataset_split(DATASET_KEY, max_samples=MAX_DATASET)
train, val, test = split_train_val_test(samples, seed=SEED)
print(f"[data] train={len(train)}  val={len(val)}  test={len(test)}")

encoders = FrozenEncoders()
student  = load_student_from_hf() if QGTP_MODE != "off" else None
qgtp     = QGTPController(mode=QGTP_MODE, fixed_rho=FIXED_RHO, student=student)
llava    = LLaVAWithQGTP(lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)

optimizer = torch.optim.AdamW(llava.trainable_parameters(),
                               lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
optim_steps_per_round = STEPS_PER_ROUND // GRAD_ACCUM
total_optim_steps     = N_ROUNDS * optim_steps_per_round
scheduler = cosine_warmup_lr(optimizer, total_optim_steps,
                              warmup_frac=WARMUP_FRAC, min_lr_ratio=LR_MIN_RATIO)
optimizer.zero_grad(set_to_none=True)

# =============================================================================
# INITIAL EVAL
# =============================================================================
print("\n[eval] initial (round 0, pre-LoRA)")
init_metrics = evaluate(llava, encoders, qgtp, test,
                         batch_size=EVAL_BATCH_SIZE, max_eval=MAX_EVAL_SAMPLES,
                         desc="round 0", max_new_tokens=MAX_NEW_TOKENS)
print(f"[round 0] test acc={init_metrics['accuracy']:.3f}  "
      f"avg_k={init_metrics['avg_kept_tokens']:.0f}")

# =============================================================================
# TRAINING LOOP
# =============================================================================
def _snapshot_lora(model):
    return {k: v.detach().clone().cpu() for k, v in model.lora_state_dict().items()}

def _restore_lora(model, sd):
    model.load_lora_state_dict(sd)

history = [{"round": 0, **init_metrics, "train_loss": None}]
rng = torch.Generator(device="cpu").manual_seed(SEED)
n_train = len(train)
best_val_acc  = -1.0; best_round = 0
best_lora_sd  = _snapshot_lora(llava); rounds_no_imp = 0
all_step_losses = []; all_step_lrs = []

for round_i in range(1, N_ROUNDS + 1):
    t0 = time.time(); losses = []
    pbar = tqdm(range(STEPS_PER_ROUND), desc=f"round {round_i:2d}/{N_ROUNDS}",
                leave=False, dynamic_ncols=True)
    for step in pbar:
        batch     = [train[i] for i in torch.randint(0, n_train, (BATCH_SIZE,), generator=rng).tolist()]
        accum_idx = step % GRAD_ACCUM
        loss = train_step(llava, encoders, qgtp, batch, optimizer,
                          grad_clip=GRAD_CLIP, grad_accum_steps=GRAD_ACCUM, accum_idx=accum_idx)
        losses.append(loss); all_step_losses.append(loss)
        cur_lr = scheduler.get_last_lr()[0]
        if accum_idx + 1 == GRAD_ACCUM:
            scheduler.step(); all_step_lrs.append(cur_lr)
        pbar.set_postfix(loss=f"{loss:.3f}", lr=f"{cur_lr:.2e}")
    pbar.close()

    train_loss = sum(losses) / len(losses)
    val_m = evaluate(llava, encoders, qgtp, val, batch_size=EVAL_BATCH_SIZE,
                     max_eval=MAX_EVAL_SAMPLES, desc=f"round {round_i:2d} val",
                     max_new_tokens=MAX_NEW_TOKENS)
    test_m = None
    if (round_i % 5 == 0) or (round_i == N_ROUNDS):
        test_m = evaluate(llava, encoders, qgtp, test, batch_size=EVAL_BATCH_SIZE,
                          max_eval=MAX_EVAL_SAMPLES, desc=f"round {round_i:2d} test",
                          max_new_tokens=MAX_NEW_TOKENS)
    history_entry = {"round": round_i, "train_loss": train_loss,
                     "lr": scheduler.get_last_lr()[0], "val": val_m}
    if test_m:
        history_entry["test"] = test_m
    history.append(history_entry)
    log_extra = f"  test_acc={test_m['accuracy']:.3f}" if test_m else ""
    print(f"[round {round_i:2d}] loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
          f"val_acc={val_m['accuracy']:.3f}  avg_k={val_m['avg_kept_tokens']:.0f}"
          f"{log_extra}  ({time.time()-t0:.0f}s)")

    if val_m["accuracy"] >= best_val_acc + EARLY_STOP_MIN_DELTA:
        best_val_acc = val_m["accuracy"]; best_round = round_i
        best_lora_sd = _snapshot_lora(llava); rounds_no_imp = 0
        print(f"           ^ new best val_acc={best_val_acc:.3f}")
    else:
        rounds_no_imp += 1
        if rounds_no_imp >= EARLY_STOP_PATIENCE:
            print(f"\n[early stop] best round {best_round} (val_acc={best_val_acc:.3f}).")
            break

# =============================================================================
# RESTORE BEST + FINAL TEST
# =============================================================================
_restore_lora(llava, best_lora_sd)
final = evaluate(llava, encoders, qgtp, test, batch_size=EVAL_BATCH_SIZE, max_eval=None,
                 desc="final test", max_new_tokens=20)
print(f"[final] test_acc={final['accuracy']:.3f} on {final['n_total']} samples")

# =============================================================================
# FIGURES
# =============================================================================
val_rounds  = [h["round"] for h in history if "val"  in h]
val_accs    = [h["val"]["accuracy"] for h in history if "val" in h]
test_rounds = [h["round"] for h in history if "test" in h]
test_accs   = [h["test"]["accuracy"] for h in history if "test" in h]

fig_loss_path = os.path.join(RESULTS_DIR, f"fig_loss_{RUN_TAG}.png")
plt.figure(figsize=(8, 4.5))
plt.plot(all_step_losses, alpha=0.4, label="per micro-batch")
if len(all_step_losses) > 20:
    w = max(10, len(all_step_losses) // 50)
    smoothed = np.convolve(all_step_losses, np.ones(w)/w, mode="valid")
    plt.plot(np.arange(len(smoothed)) + w//2, smoothed, color="red", linewidth=2,
             label=f"rolling mean (w={w})")
plt.xlabel("micro-batch step"); plt.ylabel("training loss")
plt.title(f"Training loss — {RUN_TAG}"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(fig_loss_path, dpi=110); plt.close()

fig_acc_path = os.path.join(RESULTS_DIR, f"fig_acc_{RUN_TAG}.png")
plt.figure(figsize=(8, 4.5))
plt.plot(val_rounds, val_accs, marker="o", label="val")
if test_rounds:
    plt.plot(test_rounds, test_accs, marker="s", label="test (sampled)")
plt.axvline(best_round, color="green", linestyle="--", alpha=0.6,
            label=f"best round = {best_round}")
plt.axhline(final["accuracy"], color="purple", linestyle=":", alpha=0.6,
            label=f"final test = {final['accuracy']:.3f}")
plt.xlabel("round"); plt.ylabel("accuracy")
plt.title(f"Accuracy — {RUN_TAG}"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(fig_acc_path, dpi=110); plt.close()

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================
lora_path = os.path.join(RESULTS_DIR, f"lora_{RUN_TAG}.pt")
torch.save({"config": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                        "lora_dropout": LORA_DROPOUT, "best_round": best_round,
                        "best_val_acc": best_val_acc},
            "state_dict": best_lora_sd}, lora_path)

results = {
    "setting": "direct_apply", "dataset": DATASET_KEY, "qgtp_mode": QGTP_MODE,
    "fixed_rho": FIXED_RHO if QGTP_MODE == "fixed" else None,
    "best_round": best_round, "best_val_acc": best_val_acc,
    "config": {"lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
               "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
               "steps_per_round": STEPS_PER_ROUND, "seed": SEED},
    "history": history, "final_test": final,
}
results_path = os.path.join(RESULTS_DIR, f"results_{RUN_TAG}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"[results] wrote {results_path}")

# =============================================================================
# UPLOAD
# =============================================================================
def _safe_upload(local_path, repo_filename):
    try:
        upload_file(path_or_fileobj=local_path, path_in_repo=repo_filename,
                    repo_id=HF_REPO, token=HF_TOKEN)
        print(f"  uploaded -> {HF_REPO}/{repo_filename}")
    except Exception as e:
        print(f"  FAILED ({repo_filename}): {e}")

if not OFFLINE and HF_TOKEN:
    _safe_upload(results_path,  f"results_{RUN_TAG}.json")
    _safe_upload(fig_loss_path, f"figures/fig_loss_{RUN_TAG}.png")
    _safe_upload(fig_acc_path,  f"figures/fig_acc_{RUN_TAG}.png")
    _safe_upload(lora_path,     f"lora/lora_{RUN_TAG}.pt")

# =============================================================================
# CLEANUP
# =============================================================================
del optimizer, scheduler, llava, encoders
if student: del student
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    print(f"[cleanup] GPU 0: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

print(f"\n[done] final test accuracy = {final['accuracy']:.4f}")
