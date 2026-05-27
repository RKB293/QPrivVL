import os

# ============================================================================
# HARDCODED ENVIRONMENT
# ============================================================================


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ============================================================================

import sys
import gc
import time
import math
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from huggingface_hub import upload_file

from qgtp_lib import (
    setup_hf,
    load_dataset_split, split_train_val_test, shard_for_clients,
    FrozenEncoders, load_student_from_hf,
    QGTPController, LLaVAWithQGTP,
    train_step, evaluate, _get_inner,
    cosine_warmup_lr,
    OFFLINE, HF_REPO, HF_TOKEN,
)

# =============================================================================
# CONFIG
# =============================================================================
#DATASET_KEY = os.environ.get("DATASET_KEY", "okvqa")
#QGTP_MODE   = "fixed" 
#FIXED_RHO   = float(os.environ.get("RHO_KEY", 0.5))
DATASET_KEY = "vqarad"
QGTP_MODE = "off"
FIXED_RHO = 0.5

# --- Split topology ----------------------------------------------------------
N_CLIENTS        = 5
CUT_LAYER        = 16                # split point inside Llama (7B has 32 layers)

# --- Training schedule (matches module 6) ------------------------------------
N_ROUNDS         = 20
LOCAL_STEPS      = 20
GRAD_ACCUM       = 2
BATCH_SIZE       = 16
EVAL_BATCH_SIZE  = 16
MAX_EVAL_SAMPLES = 200

# --- Optimization ------------------------------------------------------------
LR               = 1e-4
WARMUP_FRAC      = 0.05
WEIGHT_DECAY     = 0.0
GRAD_CLIP        = 1.0
LR_MIN_RATIO     = 0.05

# --- LoRA capacity -----------------------------------------------------------
LORA_R           = 32
LORA_ALPHA       = 64
LORA_DROPOUT     = 0.05

# --- Early stopping ----------------------------------------------------------
EARLY_STOP_PATIENCE  = 3
EARLY_STOP_MIN_DELTA = 0.005

# --- Misc --------------------------------------------------------------------
MAX_DATASET      = 5000
MAX_NEW_TOKENS   = 20
SEED             = 42

# --- Output ------------------------------------------------------------------
RESULTS_DIR      = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_TAG = f"split_{DATASET_KEY}_{QGTP_MODE}_cut{CUT_LAYER}"
if QGTP_MODE == "fixed":
    RUN_TAG += f"_rho{FIXED_RHO}"


# =============================================================================
# ENV BANNER
# =============================================================================
print("=" * 70)
print(f"[env] HF_HOME      = {os.environ.get('HF_HOME', '<default cache>')}")
print(f"[env] HF_TOKEN     = {'set' if os.environ.get('HF_TOKEN') else 'not set'}")
print(f"[env] OFFLINE mode = {OFFLINE}")
print(f"[env] CUDA         = {torch.cuda.is_available()} "
      f"(devices visible: {torch.cuda.device_count()})")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"[env]   GPU {i}: {name}  "
              f"({free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total)")
print("=" * 70)

setup_hf()
print(f"\n[config] SPLIT  dataset={DATASET_KEY}  qgtp={QGTP_MODE}  "
      f"clients={N_CLIENTS}  cut_layer={CUT_LAYER}")
print(f"[config] rounds={N_ROUNDS}  local_steps={LOCAL_STEPS}  "
      f"grad_accum={GRAD_ACCUM}  bs={BATCH_SIZE} (eff={BATCH_SIZE*GRAD_ACCUM})  "
      f"lr={LR}  lora_r={LORA_R}")
print(f"[config] run_tag={RUN_TAG}")

torch.manual_seed(SEED)


# =============================================================================
# 1) DATA + SHARDING
# =============================================================================
print("\n[data] loading + sharding")
samples = load_dataset_split(DATASET_KEY, max_samples=MAX_DATASET)
train, val, test = split_train_val_test(samples, seed=SEED)
client_shards = shard_for_clients(train, N_CLIENTS, seed=SEED)
print(f"[data] train={len(train)}  val={len(val)}  test={len(test)}")
for ci, sh in enumerate(client_shards):
    print(f"  client {ci}: {len(sh)} samples")


# =============================================================================
# 2) COMPONENTS + SPLIT-LEARNING INSTRUMENTATION
# =============================================================================
encoders = FrozenEncoders()
student  = load_student_from_hf() if QGTP_MODE != "off" else None
qgtp     = QGTPController(mode=QGTP_MODE, fixed_rho=FIXED_RHO, student=student)
llava    = LLaVAWithQGTP(
    lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
)

# Locate the LM and verify the cut layer is in range. We hook the cut layer
# to record the hidden state at the boundary -- this is purely a measurement;
# the model's forward is otherwise unchanged.
_lm = _get_inner(llava.llava, "language_model")
_decoder_layers = _lm.layers
_n_layers = len(_decoder_layers)
assert 0 < CUT_LAYER < _n_layers, (
    f"cut_layer {CUT_LAYER} out of range 1..{_n_layers - 1}")
print(f"[split] LM has {_n_layers} decoder layers; "
      f"cut at layer {CUT_LAYER}  "
      f"(client owns 0..{CUT_LAYER - 1}, server owns {CUT_LAYER}..{_n_layers - 1})")

_cut_state = {"input": None, "output": None}

def _record_cut_input(module, args, kwargs):
    """Pre-forward hook on the cut layer: record what the client passes
    to the server. In a real SL deployment this hidden state is what would
    be transmitted across the cut; we observe it for diagnostics."""
    h = args[0] if args else kwargs.get("hidden_states")
    _cut_state["input"] = h.detach()
    return None  # don't modify

def _record_cut_output(module, args, output):
    """Post-forward hook on the LAST decoder layer: record the activation
    that becomes the LM head input. Useful for diagnostics."""
    h = output[0] if isinstance(output, tuple) else output
    _cut_state["output"] = h.detach()
    return None

# Install the hooks. We don't actually need them for correctness, but they
# make it easy to log smashed-data shape and demonstrate the cut is real.
_h1 = _decoder_layers[CUT_LAYER].register_forward_pre_hook(
    _record_cut_input, with_kwargs=True)
_h2 = _decoder_layers[-1].register_forward_hook(_record_cut_output)
print(f"[split] hooks registered on layers {CUT_LAYER} (pre) and "
      f"{_n_layers - 1} (post)")


# =============================================================================
# 3) OPTIMIZER + SCHEDULE
# =============================================================================
optimizer = torch.optim.AdamW(
    llava.trainable_parameters(),
    lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY,
)
optim_steps_per_round = N_CLIENTS * (LOCAL_STEPS // GRAD_ACCUM)
total_optim_steps     = N_ROUNDS * optim_steps_per_round
scheduler = cosine_warmup_lr(
    optimizer, total_optim_steps,
    warmup_frac=WARMUP_FRAC, min_lr_ratio=LR_MIN_RATIO,
)
optimizer.zero_grad(set_to_none=True)
print(f"[schedule] {optim_steps_per_round} optim steps/round, "
      f"{total_optim_steps} total")


# =============================================================================
# 4) INITIAL EVAL
# =============================================================================
print("\n[eval] initial (round 0, pre-LoRA)")
init_metrics = evaluate(
    llava, encoders, qgtp, test,
    batch_size=EVAL_BATCH_SIZE,
    max_eval=MAX_EVAL_SAMPLES, desc="round 0",
    max_new_tokens=MAX_NEW_TOKENS,
)
print(f"[round 0] test acc={init_metrics['accuracy']:.3f}  "
      f"avg_k={init_metrics['avg_kept_tokens']:.0f}")
if _cut_state["input"] is not None:
    print(f"[split] smashed-data shape at cut = "
          f"{tuple(_cut_state['input'].shape)}  "
          f"({_cut_state['input'].numel() * 2 / 1024**2:.1f} MB in bf16)")


# =============================================================================
# 5) TRAINING LOOP (sequential clients, shared server)
# =============================================================================
def _snapshot_lora(model):
    return {k: v.detach().clone().cpu() for k, v in model.lora_state_dict().items()}

def _restore_lora(model, sd):
    model.load_lora_state_dict(sd)


history = [{"round": 0, **init_metrics, "train_loss": None}]
rng = torch.Generator(device="cpu").manual_seed(SEED)

best_val_acc   = -1.0
best_round     = 0
best_lora_sd   = _snapshot_lora(llava)
rounds_no_imp  = 0

all_step_losses = []
all_step_lrs    = []
global_step     = 0

for round_i in range(1, N_ROUNDS + 1):
    t0 = time.time()
    round_losses = []

    # Sequential SL: each client trains on the SHARED model in turn. There
    # is no per-client model copy and no aggregation -- the server's state
    # is updated incrementally across clients within the round.
    for ci in range(N_CLIENTS):
        shard = client_shards[ci]
        running_loss = 0.0; n_seen = 0

        pbar = tqdm(
            range(LOCAL_STEPS),
            desc=f"r{round_i:2d}/{N_ROUNDS} c{ci+1}/{N_CLIENTS}",
            leave=False, dynamic_ncols=True,
        )
        for step in pbar:
            idx = torch.randint(0, len(shard), (BATCH_SIZE,), generator=rng).tolist()
            batch = [shard[i] for i in idx]
            accum_idx = (global_step * GRAD_ACCUM + step) % GRAD_ACCUM
            # train_step performs the full forward+backward; the hooks
            # above transparently record activations at the cut.
            loss = train_step(
                llava, encoders, qgtp, batch, optimizer,
                grad_clip=GRAD_CLIP,
                grad_accum_steps=GRAD_ACCUM,
                accum_idx=accum_idx,
            )
            round_losses.append(loss)
            all_step_losses.append(loss)
            n_seen += 1
            running_loss += (loss - running_loss) / n_seen
            cur_lr = scheduler.get_last_lr()[0]

            if accum_idx + 1 == GRAD_ACCUM:
                scheduler.step()
                all_step_lrs.append(cur_lr)
                global_step += 1

            pbar.set_postfix(loss=f"{loss:.3f}",
                             avg=f"{running_loss:.3f}",
                             lr=f"{cur_lr:.2e}")
        pbar.close()

    avg_round_loss = sum(round_losses) / len(round_losses)

    # Eval (val every round, test every 5)
    val_m = evaluate(
        llava, encoders, qgtp, val,
        batch_size=EVAL_BATCH_SIZE, max_eval=MAX_EVAL_SAMPLES,
        desc=f"round {round_i:2d} val", max_new_tokens=MAX_NEW_TOKENS,
    )
    test_m = None
    log_extra = ""
    if (round_i % 5 == 0) or (round_i == N_ROUNDS):
        test_m = evaluate(
            llava, encoders, qgtp, test,
            batch_size=EVAL_BATCH_SIZE, max_eval=MAX_EVAL_SAMPLES,
            desc=f"round {round_i:2d} test", max_new_tokens=MAX_NEW_TOKENS,
        )
        log_extra = f"  test_acc={test_m['accuracy']:.3f}"

    history_entry = {
        "round": round_i,
        "avg_round_loss": avg_round_loss,
        "val": val_m,
    }
    if test_m is not None:
        history_entry["test"] = test_m
    history.append(history_entry)

    elapsed = time.time() - t0
    cur_lr  = scheduler.get_last_lr()[0]
    print(f"[round {round_i:2d}] loss={avg_round_loss:.4f}  lr={cur_lr:.2e}  "
          f"val_acc={val_m['accuracy']:.3f}  "
          f"avg_k={val_m['avg_kept_tokens']:.0f}{log_extra}  ({elapsed:.0f}s)")

    if val_m["accuracy"] >= best_val_acc + EARLY_STOP_MIN_DELTA:
        best_val_acc = val_m["accuracy"]
        best_round   = round_i
        best_lora_sd = _snapshot_lora(llava)
        rounds_no_imp = 0
        print(f"           ^ new best val_acc={best_val_acc:.3f}  (snapshot saved)")
    else:
        rounds_no_imp += 1
        if rounds_no_imp >= EARLY_STOP_PATIENCE:
            print(f"\n[early stop] no val improvement for {EARLY_STOP_PATIENCE} rounds. "
                  f"Best was round {best_round} (val_acc={best_val_acc:.3f}).")
            break


# =============================================================================
# 6) RESTORE BEST + FINAL TEST
# =============================================================================
print(f"\n[restore] rolling back to best snapshot (round {best_round}, "
      f"val_acc={best_val_acc:.3f})")
_restore_lora(llava, best_lora_sd)

# Hooks aren't needed during final eval but leave them on; they're cheap.
print("\n[eval] final on full test set")
final = evaluate(
    llava, encoders, qgtp, test,
    batch_size=EVAL_BATCH_SIZE, max_eval=None,
    desc="final test", max_new_tokens=MAX_NEW_TOKENS,
)
print(f"[final] test_acc={final['accuracy']:.3f} on {final['n_total']} samples")
if _cut_state["input"] is not None:
    print(f"[split] smashed-data tensor at cut: shape={tuple(_cut_state['input'].shape)}, "
          f"dtype={_cut_state['input'].dtype}")

# Remove hooks now that we're done
_h1.remove(); _h2.remove()


# =============================================================================
# 7) FIGURES
# =============================================================================
print("\n[figures] writing plots")
val_rounds  = [h["round"] for h in history if "val"  in h]
val_accs    = [h["val"]["accuracy"]  for h in history if "val"  in h]
test_rounds = [h["round"] for h in history if "test" in h]
test_accs   = [h["test"]["accuracy"] for h in history if "test" in h]
if history[0].get("accuracy") is not None:
    test_rounds = [0] + test_rounds
    test_accs   = [history[0]["accuracy"]] + test_accs

fig_loss_path = os.path.join(RESULTS_DIR, f"fig_loss_{RUN_TAG}.png")
plt.figure(figsize=(8, 4.5))
plt.plot(all_step_losses, alpha=0.4, label="per micro-batch")
if len(all_step_losses) > 20:
    w = max(10, len(all_step_losses) // 50)
    smoothed = np.convolve(all_step_losses, np.ones(w)/w, mode="valid")
    plt.plot(np.arange(len(smoothed)) + w//2, smoothed,
             color="red", linewidth=2, label=f"rolling mean (w={w})")
plt.xlabel("micro-batch step"); plt.ylabel("training loss")
plt.title(f"Split-learning training loss -- {RUN_TAG}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(fig_loss_path, dpi=110); plt.close()
print(f"  wrote {fig_loss_path}")

fig_acc_path = os.path.join(RESULTS_DIR, f"fig_acc_{RUN_TAG}.png")
plt.figure(figsize=(8, 4.5))
plt.plot(val_rounds, val_accs, marker="o", label="val")
if test_rounds:
    plt.plot(test_rounds, test_accs, marker="s", label="test (sampled)")
plt.axvline(best_round, color="green", linestyle="--", alpha=0.6,
            label=f"best round = {best_round}")
plt.axhline(final["accuracy"], color="purple", linestyle=":", alpha=0.6,
            label=f"final test (full) = {final['accuracy']:.3f}")
plt.xlabel("round"); plt.ylabel("accuracy")
plt.title(f"Split-learning accuracy over rounds -- {RUN_TAG}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(fig_acc_path, dpi=110); plt.close()
print(f"  wrote {fig_acc_path}")

fig_lr_path = os.path.join(RESULTS_DIR, f"fig_lr_{RUN_TAG}.png")
plt.figure(figsize=(8, 3.5))
plt.plot(all_step_lrs)
plt.xlabel("optimizer step"); plt.ylabel("learning rate")
plt.title(f"LR schedule -- {RUN_TAG}")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(fig_lr_path, dpi=110); plt.close()
print(f"  wrote {fig_lr_path}")


# =============================================================================
# 8) SAVE LoRA + RESULTS
# =============================================================================
lora_path = os.path.join(RESULTS_DIR, f"lora_{RUN_TAG}.pt")
torch.save({
    "config": {
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "best_round": best_round, "best_val_acc": best_val_acc,
        "n_clients": N_CLIENTS, "cut_layer": CUT_LAYER,
    },
    "state_dict": best_lora_sd,
}, lora_path)
print(f"[lora] saved best LoRA -> {lora_path}")

results = {
    "setting":      "split_learning",
    "dataset":      DATASET_KEY,
    "qgtp_mode":    QGTP_MODE,
    "fixed_rho":    FIXED_RHO if QGTP_MODE == "fixed" else None,
    "n_clients":    N_CLIENTS,
    "cut_layer":    CUT_LAYER,
    "n_rounds_max": N_ROUNDS,
    "n_rounds_run": history[-1]["round"],
    "best_round":   best_round,
    "best_val_acc": best_val_acc,
    "config": {
        "lr": LR, "warmup_frac": WARMUP_FRAC, "lr_min_ratio": LR_MIN_RATIO,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "local_steps": LOCAL_STEPS,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "seed": SEED,
    },
    "history":      history,
    "final_test":   final,
}
results_path = os.path.join(RESULTS_DIR, f"results_{RUN_TAG}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"[results] wrote {results_path}")


# =============================================================================
# 9) UPLOAD
# =============================================================================
def _safe_upload(local_path, repo_filename):
    try:
        upload_file(path_or_fileobj=local_path, path_in_repo=repo_filename,
                    repo_id=HF_REPO, token=HF_TOKEN)
        print(f"  uploaded -> {HF_REPO}/{repo_filename}")
    except Exception as e:
        print(f"  FAILED  ({repo_filename}): {e}")

if OFFLINE or not HF_TOKEN:
    if not OFFLINE:
        print("\n[upload] no HF_TOKEN set -- skipping Hub upload.")
    else:
        print("\n[upload] OFFLINE mode -- skipping Hub upload.")
else:
    print(f"\n[upload] pushing artifacts to {HF_REPO}")
    _safe_upload(results_path,  f"results_{RUN_TAG}.json")
    _safe_upload(fig_loss_path, f"figures/fig_loss_{RUN_TAG}.png")
    _safe_upload(fig_acc_path,  f"figures/fig_acc_{RUN_TAG}.png")
    _safe_upload(fig_lr_path,   f"figures/fig_lr_{RUN_TAG}.png")
    _safe_upload(lora_path,     f"lora/lora_{RUN_TAG}.pt")


# =============================================================================
# 10) GARBAGE COLLECTION
# =============================================================================
print("\n[cleanup] freeing GPU memory")
del optimizer, scheduler
del llava, encoders
if student is not None:
    del student
del best_lora_sd
del all_step_losses, all_step_lrs
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    print(f"[cleanup] GPU 0 after cleanup: {free/1024**3:.1f} GB free / "
          f"{total/1024**3:.1f} GB total")

print(f"\n[done] final test accuracy = {final['accuracy']:.4f} "
      f"on {final['n_total']} samples")
print(f"[done] artifacts in {RESULTS_DIR} (tag: {RUN_TAG})")

