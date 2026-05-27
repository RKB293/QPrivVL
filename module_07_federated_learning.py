"""
module_07_federated_learning.py
================================

Federated learning (FedAvg) for LLaVA-1.5-7B with LoRA under QGTP.

N_CLIENTS clients train on disjoint shards; the server aggregates via
equal-weighted parameter averaging every round.

Usage
-----
    DATASET_KEY=vqarad QGTP_MODE=off python module_07_federated_learning.py
"""

import os, gc, time, json
import torch, numpy as np
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from huggingface_hub import upload_file

from qgtp_lib import (
    setup_hf, load_dataset_split, split_train_val_test, shard_for_clients,
    FrozenEncoders, load_student_from_hf, QGTPController, LLaVAWithQGTP,
    train_step, evaluate, average_state_dicts, cosine_warmup_lr,
    OFFLINE, HF_REPO, HF_TOKEN,
)

# =============================================================================
# CONFIG
# =============================================================================
DATASET_KEY = os.environ.get("DATASET_KEY", "vqarad")
QGTP_MODE   = os.environ.get("QGTP_MODE",   "off")
FIXED_RHO   = float(os.environ.get("FIXED_RHO", "0.5"))

N_CLIENTS        = 5
N_ROUNDS         = 20
LOCAL_STEPS      = 20
GRAD_ACCUM       = 2
BATCH_SIZE       = 16
EVAL_BATCH_SIZE  = 16
MAX_EVAL_SAMPLES = 200
LR               = 1e-4
WARMUP_FRAC      = 0.05
WEIGHT_DECAY     = 0.0
GRAD_CLIP        = 1.0
LR_MIN_RATIO     = 0.05
LORA_R           = 32
LORA_ALPHA       = 64
LORA_DROPOUT     = 0.05
EARLY_STOP_PATIENCE  = 4
EARLY_STOP_MIN_DELTA = 0.005
MAX_DATASET      = 5000
MAX_NEW_TOKENS   = 20
SEED             = 42

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_TAG = f"federated_{DATASET_KEY}_{QGTP_MODE}"
if QGTP_MODE == "fixed":
    RUN_TAG += f"_rho{FIXED_RHO}"

setup_hf()
torch.manual_seed(SEED)
print(f"[config] FEDERATED  dataset={DATASET_KEY}  qgtp={QGTP_MODE}  "
      f"clients={N_CLIENTS}  rounds={N_ROUNDS}  local_steps={LOCAL_STEPS}")

# =============================================================================
# DATA + SHARDING
# =============================================================================
samples = load_dataset_split(DATASET_KEY, max_samples=MAX_DATASET)
train, val, test = split_train_val_test(samples, seed=SEED)
client_shards = shard_for_clients(train, N_CLIENTS, seed=SEED)
print(f"[data] train={len(train)}  val={len(val)}  test={len(test)}")
for ci, sh in enumerate(client_shards):
    print(f"  client {ci}: {len(sh)} samples")

# =============================================================================
# COMPONENTS
# =============================================================================
encoders = FrozenEncoders()
student  = load_student_from_hf() if QGTP_MODE != "off" else None
qgtp     = QGTPController(mode=QGTP_MODE, fixed_rho=FIXED_RHO, student=student)
llava    = LLaVAWithQGTP(lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)

global_lora_sd = {k: v.detach().clone().cpu() for k, v in llava.lora_state_dict().items()}
optim_steps_per_round = N_CLIENTS * (LOCAL_STEPS // GRAD_ACCUM)
total_optim_steps     = N_ROUNDS * optim_steps_per_round


def _make_optimizer_and_scheduler(initial_step: int):
    opt = torch.optim.AdamW(llava.trainable_parameters(),
                            lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    sch = cosine_warmup_lr(opt, total_optim_steps,
                           warmup_frac=WARMUP_FRAC, min_lr_ratio=LR_MIN_RATIO)
    sch.last_epoch = initial_step - 1
    return opt, sch


# =============================================================================
# INITIAL EVAL
# =============================================================================
print("\n[eval] initial (round 0)")
init_metrics = evaluate(llava, encoders, qgtp, test, batch_size=EVAL_BATCH_SIZE,
                         max_eval=MAX_EVAL_SAMPLES, desc="round 0",
                         max_new_tokens=MAX_NEW_TOKENS)
print(f"[round 0] test acc={init_metrics['accuracy']:.3f}  "
      f"avg_k={init_metrics['avg_kept_tokens']:.0f}")

# =============================================================================
# FEDERATED ROUNDS
# =============================================================================
def _snapshot_lora(model):
    return {k: v.detach().clone().cpu() for k, v in model.lora_state_dict().items()}

def _restore_lora(model, sd):
    model.load_lora_state_dict(sd)

history = [{"round": 0, **init_metrics, "train_loss": None}]
rng = torch.Generator(device="cpu").manual_seed(SEED)
best_val_acc = -1.0; best_round = 0
best_lora_sd = {k: v.clone() for k, v in global_lora_sd.items()}
rounds_no_imp = 0; global_optim_step = 0
all_step_losses = []; all_step_lrs = []

for round_i in range(1, N_ROUNDS + 1):
    t0 = time.time()
    client_lora_updates = []; client_avg_losses = []

    for ci in range(N_CLIENTS):
        _restore_lora(llava, global_lora_sd)
        optimizer, scheduler = _make_optimizer_and_scheduler(global_optim_step)
        optimizer.zero_grad(set_to_none=True)
        shard = client_shards[ci]; client_losses = []
        running_loss = 0.0; n_seen = 0

        pbar = tqdm(range(LOCAL_STEPS), desc=f"r{round_i:2d}/{N_ROUNDS} c{ci+1}/{N_CLIENTS}",
                    leave=False, dynamic_ncols=True)
        for step in pbar:
            batch     = [shard[i] for i in torch.randint(0, len(shard), (BATCH_SIZE,), generator=rng).tolist()]
            accum_idx = step % GRAD_ACCUM
            loss = train_step(llava, encoders, qgtp, batch, optimizer,
                              grad_clip=GRAD_CLIP, grad_accum_steps=GRAD_ACCUM, accum_idx=accum_idx)
            client_losses.append(loss); all_step_losses.append(loss)
            n_seen += 1; running_loss += (loss - running_loss) / n_seen
            cur_lr = scheduler.get_last_lr()[0]
            if accum_idx + 1 == GRAD_ACCUM:
                scheduler.step(); all_step_lrs.append(cur_lr); global_optim_step += 1
            pbar.set_postfix(loss=f"{loss:.3f}", avg=f"{running_loss:.3f}", lr=f"{cur_lr:.2e}")
        pbar.close()

        client_lora_updates.append(_snapshot_lora(llava))
        client_avg_losses.append(sum(client_losses) / len(client_losses))
        del optimizer, scheduler

    global_lora_sd = average_state_dicts(client_lora_updates)
    _restore_lora(llava, global_lora_sd)

    val_m = evaluate(llava, encoders, qgtp, val, batch_size=EVAL_BATCH_SIZE,
                     max_eval=MAX_EVAL_SAMPLES, desc=f"round {round_i:2d} val",
                     max_new_tokens=MAX_NEW_TOKENS)
    test_m = None
    if (round_i % 5 == 0) or (round_i == N_ROUNDS):
        test_m = evaluate(llava, encoders, qgtp, test, batch_size=EVAL_BATCH_SIZE,
                          max_eval=MAX_EVAL_SAMPLES, desc=f"round {round_i:2d} test",
                          max_new_tokens=MAX_NEW_TOKENS)

    avg_round_loss = sum(client_avg_losses) / len(client_avg_losses)
    history_entry  = {"round": round_i, "avg_round_loss": avg_round_loss, "val": val_m}
    if test_m:
        history_entry["test"] = test_m
    history.append(history_entry)
    log_extra = f"  test_acc={test_m['accuracy']:.3f}" if test_m else ""
    print(f"[round {round_i:2d}] avg_loss={avg_round_loss:.4f}  "
          f"val_acc={val_m['accuracy']:.3f}  avg_k={val_m['avg_kept_tokens']:.0f}"
          f"{log_extra}  ({time.time()-t0:.0f}s)")

    if val_m["accuracy"] >= best_val_acc + EARLY_STOP_MIN_DELTA:
        best_val_acc = val_m["accuracy"]; best_round = round_i
        best_lora_sd = {k: v.clone() for k, v in global_lora_sd.items()}
        rounds_no_imp = 0
        print(f"           ^ new best val_acc={best_val_acc:.3f}")
    else:
        rounds_no_imp += 1
        if rounds_no_imp >= EARLY_STOP_PATIENCE:
            print(f"\n[early stop] best round {best_round} (val_acc={best_val_acc:.3f}).")
            break

# =============================================================================
# RESTORE + FINAL TEST
# =============================================================================
_restore_lora(llava, best_lora_sd)
final = evaluate(llava, encoders, qgtp, test, batch_size=EVAL_BATCH_SIZE, max_eval=None,
                 desc="final test", max_new_tokens=MAX_NEW_TOKENS)
print(f"[final] test_acc={final['accuracy']:.3f} on {final['n_total']} samples")

# =============================================================================
# FIGURES
# =============================================================================
val_rounds = [h["round"] for h in history if "val" in h]
val_accs   = [h["val"]["accuracy"] for h in history if "val" in h]
test_rounds = [h["round"] for h in history if "test" in h]
test_accs   = [h["test"]["accuracy"] for h in history if "test" in h]

fig_loss_path = os.path.join(RESULTS_DIR, f"fig_loss_{RUN_TAG}.png")
plt.figure(figsize=(8, 4.5))
plt.plot(all_step_losses, alpha=0.4, label="per micro-batch (all clients)")
if len(all_step_losses) > 20:
    w = max(10, len(all_step_losses) // 50)
    smoothed = np.convolve(all_step_losses, np.ones(w)/w, mode="valid")
    plt.plot(np.arange(len(smoothed)) + w//2, smoothed, color="red", linewidth=2,
             label=f"rolling mean (w={w})")
plt.xlabel("micro-batch step"); plt.ylabel("training loss")
plt.title(f"FL training loss — {RUN_TAG}"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(fig_loss_path, dpi=110); plt.close()

fig_acc_path = os.path.join(RESULTS_DIR, f"fig_acc_{RUN_TAG}.png")
plt.figure(figsize=(8, 4.5))
plt.plot(val_rounds, val_accs, marker="o", label="val (aggregated model)")
if test_rounds:
    plt.plot(test_rounds, test_accs, marker="s", label="test (sampled)")
plt.axvline(best_round, color="green", linestyle="--", alpha=0.6, label=f"best round = {best_round}")
plt.axhline(final["accuracy"], color="purple", linestyle=":", alpha=0.6,
            label=f"final test = {final['accuracy']:.3f}")
plt.xlabel("FL round"); plt.ylabel("accuracy"); plt.title(f"FL accuracy — {RUN_TAG}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(fig_acc_path, dpi=110); plt.close()

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================
lora_path = os.path.join(RESULTS_DIR, f"lora_{RUN_TAG}.pt")
torch.save({"config": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "n_clients": N_CLIENTS,
                        "best_round": best_round, "best_val_acc": best_val_acc},
            "state_dict": best_lora_sd}, lora_path)

results = {
    "setting": "federated", "dataset": DATASET_KEY, "qgtp_mode": QGTP_MODE,
    "fixed_rho": FIXED_RHO if QGTP_MODE == "fixed" else None,
    "n_clients": N_CLIENTS, "best_round": best_round, "best_val_acc": best_val_acc,
    "config": {"lr": LR, "lora_r": LORA_R, "n_clients": N_CLIENTS,
               "local_steps": LOCAL_STEPS, "seed": SEED},
    "history": history, "final_test": final,
}
results_path = os.path.join(RESULTS_DIR, f"results_{RUN_TAG}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"[results] wrote {results_path}")

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

del llava, encoders
if student: del student
del best_lora_sd, global_lora_sd
gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f"\n[done] final test accuracy = {final['accuracy']:.4f}")
