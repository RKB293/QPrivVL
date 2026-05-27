import os

# ============================================================================
# HARDCODED ENVIRONMENT
# ============================================================================


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ============================================================================

import sys
import gc
import re
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
DATASET_KEY = os.environ.get("DATASET_KEY", "okvqa")
QGTP_MODE   = os.environ.get("MODE_KEY", "fixed")
FIXED_RHO   = float(os.environ.get("RHO_KEY", 0.5))
# --- U-shape topology --------------------------------------------------------
N_CLIENTS        = 5
CUT_A            = 8
CUT_B            = 24

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

RUN_TAG = f"ushape_{DATASET_KEY}_{QGTP_MODE}_a{CUT_A}b{CUT_B}"
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
print(f"\n[config] U-SHAPE  dataset={DATASET_KEY}  qgtp={QGTP_MODE}  "
      f"clients={N_CLIENTS}  cut_a={CUT_A}  cut_b={CUT_B}")
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
# 2) COMPONENTS + U-SHAPE INSTRUMENTATION
# =============================================================================
encoders = FrozenEncoders()
student  = load_student_from_hf() if QGTP_MODE != "off" else None
qgtp     = QGTPController(mode=QGTP_MODE, fixed_rho=FIXED_RHO, student=student)
llava    = LLaVAWithQGTP(
    lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
)

_lm = _get_inner(llava.llava, "language_model")
_decoder_layers = _lm.layers
_n_layers = len(_decoder_layers)
assert 0 < CUT_A < CUT_B < _n_layers, (
    f"cuts ({CUT_A}, {CUT_B}) must satisfy 0 < CUT_A < CUT_B < {_n_layers}")
print(f"[u-shape] LM has {_n_layers} decoder layers; "
      f"client-A=0..{CUT_A - 1}  server={CUT_A}..{CUT_B - 1}  "
      f"client-B={CUT_B}..{_n_layers - 1}")

# Two diagnostic hooks: one at each cut.
_cut_state = {"to_server": None, "from_server": None}

def _record_to_server(module, args, kwargs):
    h = args[0] if args else kwargs.get("hidden_states")
    _cut_state["to_server"] = h.detach()

def _record_from_server(module, args, kwargs):
    h = args[0] if args else kwargs.get("hidden_states")
    _cut_state["from_server"] = h.detach()

_h1 = _decoder_layers[CUT_A].register_forward_pre_hook(
    _record_to_server, with_kwargs=True)
_h2 = _decoder_layers[CUT_B].register_forward_pre_hook(
    _record_from_server, with_kwargs=True)
print(f"[u-shape] hooks registered at layers {CUT_A} (client->server) "
      f"and {CUT_B} (server->client)")


# =============================================================================
# 2b) PARTITION TRAINABLE PARAMS: SERVER-SHARED vs CLIENT-PRIVATE
# =============================================================================
# Layer-index regex covers Llama decoder param names like
# "...language_model.layers.<i>.self_attn.q_proj.lora_A.default.weight".
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

def _layer_index_from_name(name: str):
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None

def _is_server_side(name: str) -> bool:
    """True iff the parameter belongs to a transformer layer in [CUT_A, CUT_B).
    Non-layer params (projector, embeddings, lm_head, etc.) are client-side."""
    li = _layer_index_from_name(name)
    if li is None:
        return False
    return CUT_A <= li < CUT_B

named_trainable = [(n, p) for n, p in llava.named_parameters() if p.requires_grad]
server_named = [(n, p) for n, p in named_trainable if _is_server_side(n)]
client_named = [(n, p) for n, p in named_trainable if not _is_server_side(n)]
server_params = [p for _, p in server_named]
client_params = [p for _, p in client_named]
print(f"[u-shape] trainable tensors: "
      f"server-side (shared) = {len(server_named)}, "
      f"client-side (per-client) = {len(client_named)}")


# =============================================================================
# 3) OPTIMIZERS + SCHEDULES (server-shared, client-private)
# =============================================================================
# One server optimizer (shared) + N client optimizers (private state).
# All optimizers reference the SAME live parameter tensors -- the
# parameter VALUES swap via client_side_state below; the optimizer
# MOMENTS (m, v) live inside each optimizer's .state dict and are
# naturally per-client because we only ever call .step() on the
# current client's optimizer.
server_optimizer = torch.optim.AdamW(
    server_params, lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY,
)
client_optimizers = [
    torch.optim.AdamW(
        client_params, lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY,
    )
    for _ in range(N_CLIENTS)
]

# Per-client snapshots of the *client-side* trainable weights only.
# Kept on-device (no .cpu()) -- these are small LoRA tensors.
def _snapshot_client_side():
    return {n: p.detach().clone() for n, p in client_named}

def _restore_client_side(snapshot):
    with torch.no_grad():
        for n, p in client_named:
            p.copy_(snapshot[n])

# Initialize all clients to the SAME starting client-side weights, so the
# only source of divergence is each client's own data shard.
_initial_client_side = _snapshot_client_side()
client_side_state = [
    {n: t.clone() for n, t in _initial_client_side.items()}
    for _ in range(N_CLIENTS)
]

# Schedules: shared server schedule sees N_CLIENTS * (LOCAL_STEPS//GRAD_ACCUM)
# steps per round; each client's schedule only ticks during that client's
# turn -> (LOCAL_STEPS // GRAD_ACCUM) per round per client.
optim_steps_per_round = N_CLIENTS * (LOCAL_STEPS // GRAD_ACCUM)
total_optim_steps     = N_ROUNDS * optim_steps_per_round
server_scheduler = cosine_warmup_lr(
    server_optimizer, total_optim_steps,
    warmup_frac=WARMUP_FRAC, min_lr_ratio=LR_MIN_RATIO,
)
client_total_steps = N_ROUNDS * (LOCAL_STEPS // GRAD_ACCUM)
client_schedulers = [
    cosine_warmup_lr(
        opt, client_total_steps,
        warmup_frac=WARMUP_FRAC, min_lr_ratio=LR_MIN_RATIO,
    )
    for opt in client_optimizers
]

# Multi-optimizer shim so train_step (which expects a single optimizer
# object with .zero_grad / .step / .param_groups) can drive both at once.
class _MultiOptim:
    def __init__(self, opts):
        self.opts = opts
    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)
    def step(self, closure=None):
        for o in self.opts:
            o.step()
    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]
    @property
    def state(self):
        # Aggregated read-only view (rarely used by train_step, but provided
        # for compatibility with utilities that probe optimizer state).
        merged = {}
        for o in self.opts:
            merged.update(o.state)
        return merged

server_optimizer.zero_grad(set_to_none=True)
for o in client_optimizers:
    o.zero_grad(set_to_none=True)
print(f"[schedule] {optim_steps_per_round} optim steps/round (server), "
      f"{client_total_steps} optim steps total per client")


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
if _cut_state["to_server"] is not None:
    s_in  = tuple(_cut_state["to_server"].shape)
    s_out = tuple(_cut_state["from_server"].shape)
    sz_mb = _cut_state["to_server"].numel() * 2 / 1024**2
    print(f"[u-shape] activation shapes -- to-server={s_in}  "
          f"from-server={s_out}  (~{sz_mb:.1f} MB each, bf16)")


# =============================================================================
# 5) TRAINING LOOP (sequential clients, shared server, private client halves)
# =============================================================================
def _snapshot_lora(model):
    """Full LoRA snapshot (used for the 'last live' state on disk).
    For the *best* model we snapshot per-client state separately below."""
    return {k: v.detach().clone().cpu() for k, v in model.lora_state_dict().items()}

def _restore_lora(model, sd):
    model.load_lora_state_dict(sd)

def _snapshot_server_side_cpu():
    """CPU snapshot of just the server-side trainable tensors."""
    return {n: p.detach().clone().cpu() for n, p in server_named}

def _snapshot_all_client_side_cpu():
    """CPU snapshot of every client's private client-side state."""
    return [
        {n: t.detach().clone().cpu() for n, t in cs.items()}
        for cs in client_side_state
    ]

def _restore_server_side(sd_cpu):
    with torch.no_grad():
        for n, p in server_named:
            p.copy_(sd_cpu[n].to(p.device, dtype=p.dtype))


history = [{"round": 0, **init_metrics, "train_loss": None}]
rng = torch.Generator(device="cpu").manual_seed(SEED)

best_val_acc        = -1.0
best_round          = 0
best_server_sd      = _snapshot_server_side_cpu()
best_client_sds     = _snapshot_all_client_side_cpu()
best_eval_client    = 0   # which client's weights produced best_val_acc
rounds_no_imp       = 0

all_step_losses = []
all_step_lrs    = []
global_step     = 0

# Track which client's weights are currently loaded in the live model.
_current_loaded_client = None

def _swap_in_client(ci):
    """Load client ci's private client-side weights into the live model."""
    global _current_loaded_client
    if _current_loaded_client == ci:
        return
    if _current_loaded_client is not None:
        # Save whatever is currently live back to its owning client.
        client_side_state[_current_loaded_client] = _snapshot_client_side()
    _restore_client_side(client_side_state[ci])
    _current_loaded_client = ci

def _sync_live_back_to_owner():
    """Persist any in-flight updates from the live model into the owning
    client's snapshot dict. Call before eval/snapshotting."""
    global _current_loaded_client
    if _current_loaded_client is not None:
        client_side_state[_current_loaded_client] = _snapshot_client_side()


for round_i in range(1, N_ROUNDS + 1):
    t0 = time.time()
    round_losses = []

    for ci in range(N_CLIENTS):
        # Swap this client's private client-side LoRA into the live model.
        _swap_in_client(ci)
        client_opt   = client_optimizers[ci]
        client_sched = client_schedulers[ci]
        multi_opt    = _MultiOptim([client_opt, server_optimizer])

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
            # train_step drives forward+backward+optimizer.step via multi_opt,
            # which steps BOTH this client's optimizer and the server optimizer.
            loss = train_step(
                llava, encoders, qgtp, batch, multi_opt,
                grad_clip=GRAD_CLIP,
                grad_accum_steps=GRAD_ACCUM,
                accum_idx=accum_idx,
            )
            round_losses.append(loss)
            all_step_losses.append(loss)
            n_seen += 1
            running_loss += (loss - running_loss) / n_seen
            cur_lr = server_scheduler.get_last_lr()[0]

            if accum_idx + 1 == GRAD_ACCUM:
                client_sched.step()
                server_scheduler.step()
                all_step_lrs.append(cur_lr)
                global_step += 1

            pbar.set_postfix(loss=f"{loss:.3f}",
                             avg=f"{running_loss:.3f}",
                             lr=f"{cur_lr:.2e}")
        pbar.close()

        # Persist this client's freshly-trained client-side weights.
        client_side_state[ci] = _snapshot_client_side()
        _current_loaded_client = ci

    avg_round_loss = sum(round_losses) / len(round_losses)

    # ---- Eval: average accuracy across all N clients' client-side weights.
    # Server-side LoRA is shared, so the only thing that changes between
    # eval passes is which client's private client-A/client-B weights are
    # loaded. We rotate through them and average.
    _sync_live_back_to_owner()
    per_client_val_accs = []
    per_client_val_metrics = []
    for ci in range(N_CLIENTS):
        _swap_in_client(ci)
        m = evaluate(
            llava, encoders, qgtp, val,
            batch_size=EVAL_BATCH_SIZE, max_eval=MAX_EVAL_SAMPLES,
            desc=f"round {round_i:2d} val c{ci+1}",
            max_new_tokens=MAX_NEW_TOKENS,
        )
        per_client_val_accs.append(m["accuracy"])
        per_client_val_metrics.append(m)
    # Aggregate val metrics. Keep the per-client accuracies for diagnostics
    # and synthesize a representative dict shaped like a single metrics dict.
    avg_val_acc = float(np.mean(per_client_val_accs))
    avg_kept = float(np.mean([m["avg_kept_tokens"] for m in per_client_val_metrics]))
    val_m = {
        "accuracy": avg_val_acc,
        "avg_kept_tokens": avg_kept,
        "per_client_accuracy": per_client_val_accs,
        "n_total": per_client_val_metrics[0].get("n_total"),
    }
    # Identify the best-performing client at this round (for snapshot eval).
    best_eval_client_this_round = int(np.argmax(per_client_val_accs))

    test_m = None
    log_extra = ""
    if (round_i % 5 == 0) or (round_i == N_ROUNDS):
        per_client_test_accs = []
        per_client_test_metrics = []
        for ci in range(N_CLIENTS):
            _swap_in_client(ci)
            tm = evaluate(
                llava, encoders, qgtp, test,
                batch_size=EVAL_BATCH_SIZE, max_eval=MAX_EVAL_SAMPLES,
                desc=f"round {round_i:2d} test c{ci+1}",
                max_new_tokens=MAX_NEW_TOKENS,
            )
            per_client_test_accs.append(tm["accuracy"])
            per_client_test_metrics.append(tm)
        test_m = {
            "accuracy": float(np.mean(per_client_test_accs)),
            "avg_kept_tokens": float(np.mean(
                [m["avg_kept_tokens"] for m in per_client_test_metrics])),
            "per_client_accuracy": per_client_test_accs,
            "n_total": per_client_test_metrics[0].get("n_total"),
        }
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
    cur_lr  = server_scheduler.get_last_lr()[0]
    spread = max(per_client_val_accs) - min(per_client_val_accs)
    print(f"[round {round_i:2d}] loss={avg_round_loss:.4f}  lr={cur_lr:.2e}  "
          f"val_acc(avg)={val_m['accuracy']:.3f}  "
          f"(spread={spread:.3f})  "
          f"avg_k={val_m['avg_kept_tokens']:.0f}{log_extra}  ({elapsed:.0f}s)")

    if val_m["accuracy"] >= best_val_acc + EARLY_STOP_MIN_DELTA:
        best_val_acc      = val_m["accuracy"]
        best_round        = round_i
        # Snapshot the FULL system: shared server-side + all N client-side
        # weights. This lets us restore an exactly-equivalent U-shape state.
        _sync_live_back_to_owner()
        best_server_sd    = _snapshot_server_side_cpu()
        best_client_sds   = _snapshot_all_client_side_cpu()
        best_eval_client  = best_eval_client_this_round
        rounds_no_imp     = 0
        print(f"           ^ new best avg val_acc={best_val_acc:.3f}  "
              f"(snapshot saved; best single client = {best_eval_client})")
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
      f"avg val_acc={best_val_acc:.3f})")
# Restore shared server-side weights.
_restore_server_side(best_server_sd)
# Restore per-client snapshots into client_side_state (on device).
_dev = client_named[0][1].device
for ci in range(N_CLIENTS):
    client_side_state[ci] = {
        n: t.to(_dev).clone() for n, t in best_client_sds[ci].items()
    }
_current_loaded_client = None  # force reload on next swap_in

# Final eval: average over all N clients on the full test set.
print("\n[eval] final on full test set (averaged over all clients)")
per_client_final_accs = []
per_client_final_metrics = []
for ci in range(N_CLIENTS):
    _swap_in_client(ci)
    fm = evaluate(
        llava, encoders, qgtp, test,
        batch_size=EVAL_BATCH_SIZE, max_eval=None,
        desc=f"final test c{ci+1}", max_new_tokens=MAX_NEW_TOKENS,
    )
    per_client_final_accs.append(fm["accuracy"])
    per_client_final_metrics.append(fm)
    print(f"  client {ci}: test_acc={fm['accuracy']:.4f} on {fm['n_total']} samples")

final = {
    "accuracy": float(np.mean(per_client_final_accs)),
    "avg_kept_tokens": float(np.mean(
        [m["avg_kept_tokens"] for m in per_client_final_metrics])),
    "per_client_accuracy": per_client_final_accs,
    "n_total": per_client_final_metrics[0].get("n_total"),
}
print(f"[final] avg test_acc={final['accuracy']:.4f}  "
      f"(min={min(per_client_final_accs):.3f}  "
      f"max={max(per_client_final_accs):.3f})  "
      f"on {final['n_total']} samples per client")

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
plt.title(f"U-shape training loss -- {RUN_TAG}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(fig_loss_path, dpi=110); plt.close()
print(f"  wrote {fig_loss_path}")

fig_acc_path = os.path.join(RESULTS_DIR, f"fig_acc_{RUN_TAG}.png")
plt.figure(figsize=(8, 4.5))
plt.plot(val_rounds, val_accs, marker="o", label="val (avg over clients)")
if test_rounds:
    plt.plot(test_rounds, test_accs, marker="s", label="test (sampled, avg)")
plt.axvline(best_round, color="green", linestyle="--", alpha=0.6,
            label=f"best round = {best_round}")
plt.axhline(final["accuracy"], color="purple", linestyle=":", alpha=0.6,
            label=f"final test (full, avg) = {final['accuracy']:.3f}")
plt.xlabel("round"); plt.ylabel("accuracy")
plt.title(f"U-shape accuracy over rounds -- {RUN_TAG}")
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
# Save the full U-shape state: shared server + per-client client-side.
torch.save({
    "config": {
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "best_round": best_round, "best_val_acc": best_val_acc,
        "n_clients": N_CLIENTS, "cut_a": CUT_A, "cut_b": CUT_B,
        "best_eval_client": best_eval_client,
    },
    "server_state_dict": best_server_sd,
    "client_state_dicts": best_client_sds,
}, lora_path)
print(f"[lora] saved best U-shape state -> {lora_path}")

results = {
    "setting":      "u_shaped_split",
    "dataset":      DATASET_KEY,
    "qgtp_mode":    QGTP_MODE,
    "fixed_rho":    FIXED_RHO if QGTP_MODE == "fixed" else None,
    "n_clients":    N_CLIENTS,
    "cut_a":        CUT_A,
    "cut_b":        CUT_B,
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
del server_optimizer, client_optimizers, server_scheduler, client_schedulers
del llava, encoders
if student is not None:
    del student
del best_server_sd, best_client_sds, client_side_state
del all_step_losses, all_step_lrs
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    print(f"[cleanup] GPU 0 after cleanup: {free/1024**3:.1f} GB free / "
          f"{total/1024**3:.1f} GB total")

print(f"\n[done] final avg test accuracy = {final['accuracy']:.4f} "
      f"on {final['n_total']} samples per client")
print(f"[done] artifacts in {RESULTS_DIR} (tag: {RUN_TAG})")
