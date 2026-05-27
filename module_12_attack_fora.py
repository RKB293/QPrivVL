"""
module_12_attack_fora.py
========================

FORA: Feature Oracle Reconstruction Attack.

Threat model
------------
The attacker is stronger than FSHA: they train an inversion decoder on
UNPRUNED smashed activations (i.e. as if the system had no QGTP defense),
then deploy that decoder against the actually-defended target activations.
This is the "oracle attacker" -- the decoder has seen the strongest
possible signal during training. It establishes an upper bound on what an
attacker could ever recover from the deployed system.

Difference from FSHA
--------------------
FSHA: train on defended public, eval on defended private.
FORA: train ONCE on UNPRUNED public, eval on defended private under each rho.

So FORA fits one feature decoder + one pixel decoder total, then evaluates
both across all defense settings. Cheaper than FSHA, but the curves are
the more conservative (= more pessimistic for the defender) measurement.

Loads
-----
- LLaVA-1.5-7B + LoRA from <MODEL_TAG> on HF
- Student model from HF
- Dataset (parsed from MODEL_TAG)

Outputs
-------
- results_fora_<MODEL_TAG>.json
- pix_decoder_fora_<MODEL_TAG>.pt
- figures/fig_fora_feature_<MODEL_TAG>.pdf
- figures/fig_fora_pixel_<MODEL_TAG>.pdf
- figures/fig_fora_qualitative_<MODEL_TAG>.pdf
"""

import os

# ============================================================================
# HARDCODED ENVIRONMENT
# ============================================================================


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.getLogger("torchao").setLevel(logging.ERROR)
# ============================================================================

import gc
import json
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
warnings.filterwarnings("ignore",
    message=".*lr_scheduler.step.*before.*optimizer.step.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from huggingface_hub import upload_file

from qgtp_lib import (
    setup_hf, FrozenEncoders, load_student_from_hf, LLaVAWithQGTP,
    HF_REPO, HF_TOKEN, OFFLINE,
)

from module_10_attack_common import (
    DEVICE, CUT_LAYER, PIXEL_SIZE, PUBLIC_SIZE, PRIVATE_SIZE, load_lora_for_tag,
    parse_dataset_from_tag, public_private_split, load_dataset_with_attribute,
    SmashedExtractor, get_or_extract_smashed,
    FeatureDecoder, PixelDecoder,
    pad_batch, get_clip_patches_for_idx, get_pixel_targets_for_idx,
    build_settings, setting_key, setting_label,
    free_cuda, gpu_status,
)

# =============================================================================
# CONFIG
# =============================================================================
'''
direct_vqarad_off
  direct_vqarad_student
  direct_slake_off
  direct_slake_student

  # ---------- Federated learning (FedAvg) ----------
  federated_vqarad_off
  federated_vqarad_student
  federated_slake_off
  federated_slake_student

  # ---------- Split learning (sequential SL, cut at LM layer 16) ----------
  split_vqarad_off_cut16
  split_vqarad_student_cut16
  split_slake_off_cut16
  split_slake_student_cut16

  # ---------- U-shaped split learning (cuts at layers 8 and 24) ----------
  ushape_vqarad_off_a8b24
  ushape_vqarad_student_a8b24
  ushape_slake_off_a8b24
  ushape_slake_student_a8b24

'''

#MODEL_TAG = os.environ.get("MODEL_TAG", "direct_vqarad_off")
MODEL_TAG = "ushape_slake_student_a8b24"


LORA_R          = 32
LORA_ALPHA      = 64
LORA_DROPOUT    = 0.05

RHO_SWEEP       = [0.0, 0.3, 0.5, 0.7, 0.9]
INCLUDE_STUDENT = True

ATTACK_EPOCHS   = 6
ATTACK_BATCH    = 16
ATTACK_LR       = 1e-4
SEED            = 42

RESULTS_DIR     = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CACHE_DIR       = os.path.join(RESULTS_DIR, "smashed_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
RUN_TAG         = f"fora_{MODEL_TAG}"


# =============================================================================
# BANNER
# =============================================================================
print("=" * 70)
print(f"[attack ] FORA -- oracle attacker trained on UNPRUNED smashed")
print(f"[model  ] {MODEL_TAG}")
print(f"[cut    ] layer {CUT_LAYER}")
print(f"[sweep  ] {RHO_SWEEP} + student={INCLUDE_STUDENT}")
print(f"[gpu    ] {gpu_status()}")
print("=" * 70)
setup_hf()
torch.manual_seed(SEED); np.random.seed(SEED)


# =============================================================================
# 1) DATA + MODEL LOADING
# =============================================================================
print("\n[data] loading dataset")
DATASET_KEY = parse_dataset_from_tag(MODEL_TAG)
samples, answer_types = load_dataset_with_attribute(
    DATASET_KEY, PUBLIC_SIZE + PRIVATE_SIZE + 200)
print(f"[data] dataset='{DATASET_KEY}', got {len(samples)} samples")

public_idx, private_idx = public_private_split(
    len(samples), PUBLIC_SIZE, PRIVATE_SIZE, seed=SEED)
print(f"[data] public={len(public_idx)}  private={len(private_idx)}")

print("\n[model] loading LLaVA + LoRA + student")
encoders = FrozenEncoders()
student  = load_student_from_hf()
llava    = LLaVAWithQGTP(lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                          lora_dropout=LORA_DROPOUT)
load_lora_for_tag(llava, MODEL_TAG, results_dir=RESULTS_DIR)
llava.llava.eval()


# =============================================================================
# 2) SMASHED EXTRACTION
# =============================================================================
print("\n[smashed] extracting under each setting (cache will be reused)")
SETTINGS = build_settings(RHO_SWEEP, include_student=INCLUDE_STUDENT)
extractor = SmashedExtractor(llava, encoders, student=student,
                              cut_layer=CUT_LAYER)
smashed_by_setting = {}
for qgtp_mode, fixed_rho in SETTINGS:
    key = setting_key(qgtp_mode, fixed_rho)
    smashed_by_setting[key] = get_or_extract_smashed(
        extractor, samples, public_idx, private_idx,
        CACHE_DIR, MODEL_TAG, qgtp_mode, fixed_rho)
extractor.close()


# =============================================================================
# 3) RECONSTRUCTION TARGETS
# =============================================================================
print("\n[targets] CLIP patch grids + pixel images")
clip_pub  = get_clip_patches_for_idx(encoders, samples, public_idx)
clip_priv = get_clip_patches_for_idx(encoders, samples, private_idx)
pix_pub   = get_pixel_targets_for_idx(samples, public_idx)
pix_priv  = get_pixel_targets_for_idx(samples, private_idx)

print(f"[free] before: {gpu_status()}")
del encoders, student, llava
free_cuda()
print(f"[free] after:  {gpu_status()}")


# =============================================================================
# 4) HELPERS
# =============================================================================
def fit_decoder(decoder, smashed_pub, target_pub, desc):
    decoder = decoder.to(DEVICE).train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=ATTACK_LR)
    n = len(smashed_pub)
    for ep in range(ATTACK_EPOCHS):
        perm = np.random.permutation(n)
        pbar = tqdm(range(0, n, ATTACK_BATCH),
                    desc=f"{desc} ep{ep+1}/{ATTACK_EPOCHS}", leave=False)
        for s in pbar:
            ids = perm[s:s + ATTACK_BATCH].tolist()
            sm  = [smashed_pub[i] for i in ids]
            sm_pad, mask = pad_batch(sm)
            tgt = target_pub[ids].to(DEVICE)
            pred = decoder(sm_pad.to(DEVICE), mask.to(DEVICE))
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    decoder.eval()
    return decoder


@torch.no_grad()
def eval_decoder(decoder, smashed_priv, target_priv):
    n = len(smashed_priv)
    psnrs, sims, mses = [], [], []
    for s in range(0, n, ATTACK_BATCH):
        sm  = smashed_priv[s:s + ATTACK_BATCH]
        sm_pad, mask = pad_batch(sm)
        tgt = target_priv[s:s + ATTACK_BATCH].to(DEVICE)
        pred = decoder(sm_pad.to(DEVICE), mask.to(DEVICE))
        m = F.mse_loss(pred, tgt, reduction="none").mean(
            dim=tuple(range(1, pred.ndim))).clamp(min=1e-12)
        psnrs.extend((-10.0 * torch.log10(m)).cpu().tolist())
        mses.extend(m.cpu().tolist())
        a, b = pred.flatten(1), tgt.flatten(1)
        sims.extend(F.cosine_similarity(a, b, dim=-1).cpu().tolist())
    return {
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std":  float(np.std(psnrs)),
        "cos_mean":  float(np.mean(sims)),
        "cos_std":   float(np.std(sims)),
        "mse_mean":  float(np.mean(mses)),
    }


# =============================================================================
# 5) FORA: train ONCE on UNPRUNED public, eval across all settings
# =============================================================================
sm_off_pub = smashed_by_setting["off"]["public"]["smashed"]

print("\n[fora-train] feature decoder on UNPRUNED public smashed")
fora_feat = fit_decoder(FeatureDecoder(), sm_off_pub, clip_pub,
                         desc="fora-feat")
print("\n[fora-train] pixel decoder on UNPRUNED public smashed")
fora_pix  = fit_decoder(PixelDecoder(), sm_off_pub, pix_pub,
                         desc="fora-pix")

# Save the pix decoder weights for the qualitative figure (need them after
# evaluation rounds free CUDA buffers).
_pix_state = {k: v.detach().cpu().clone() for k, v in fora_pix.state_dict().items()}


# =============================================================================
# 6) EVAL ACROSS DEFENSE SETTINGS
# =============================================================================
print("\n[run] FORA evaluation across defenses")
all_results = {}
for qgtp_mode, fixed_rho in SETTINGS:
    key   = setting_key(qgtp_mode, fixed_rho)
    label = setting_label(qgtp_mode, fixed_rho)
    print(f"\n--- {label} ---")
    sm_priv = smashed_by_setting[key]["private"]["smashed"]
    rhos    = smashed_by_setting[key]["private"]["rhos"]
    n_kept  = smashed_by_setting[key]["private"]["n_kept"]

    feat_metrics = eval_decoder(fora_feat, sm_priv, clip_priv)
    pix_metrics  = eval_decoder(fora_pix,  sm_priv, pix_priv)

    all_results[key] = {
        "label":         label,
        "qgtp_mode":     qgtp_mode,
        "fixed_rho":     fixed_rho,
        "rho_mean":      float(np.mean(rhos)),
        "rho_std":       float(np.std(rhos)),
        "n_kept_mean":   float(np.mean(n_kept)),
        "feature":       feat_metrics,
        "pixel":         pix_metrics,
    }
    print(f"   rho_mean={np.mean(rhos):.3f}  n_kept={np.mean(n_kept):.0f}/576")
    print(f"   feature: psnr={feat_metrics['psnr_mean']:.2f}  cos={feat_metrics['cos_mean']:.3f}")
    print(f"   pixel  : psnr={pix_metrics['psnr_mean']:.2f}  cos={pix_metrics['cos_mean']:.3f}")

del fora_feat
free_cuda()


# =============================================================================
# 7) FIGURES
# =============================================================================
print("\n[figures] writing plots")
ordered_keys = sorted(all_results.keys(), key=lambda k: all_results[k]["rho_mean"])
xs = [all_results[k]["rho_mean"] for k in ordered_keys]
student_rho = (all_results["student"]["rho_mean"]
               if "student" in all_results else None)


def _plot_curve(ax, target_kind, title):
    psnr = [all_results[k][target_kind]["psnr_mean"] for k in ordered_keys]
    cos  = [all_results[k][target_kind]["cos_mean"]  for k in ordered_keys]
    ax2 = ax.twinx()
    l1, = ax.plot(xs, psnr, marker="o", color="#d62728", label="PSNR (dB)")
    l2, = ax2.plot(xs, cos, marker="s", color="#1f77b4", label="cos sim",
                   linestyle="--")
    ax.set_xlabel(r"average $\rho$")
    ax.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("cosine similarity")
    ax.grid(alpha=0.3)
    handles, labels_ = [l1, l2], [l1.get_label(), l2.get_label()]
    if student_rho is not None:
        ax.axvline(student_rho, color="green", linestyle=":", alpha=0.7)
        handles.append(plt.Line2D([0], [0], color="green", linestyle=":"))
        labels_.append(rf"student $\rho$={student_rho:.2f}")
    ax.legend(handles=handles, labels=labels_, loc="best", fontsize=8)
    ax.set_title(title)


fig_feat_path = os.path.join(RESULTS_DIR, f"fig_fora_feature_{MODEL_TAG}.pdf")
fig, ax = plt.subplots(figsize=(6.5, 4.0))
_plot_curve(ax, "feature",
            f"FORA feature-space (oracle attacker)\n(model: {MODEL_TAG})")
plt.tight_layout(); plt.savefig(fig_feat_path); plt.close()
print(f"  wrote {fig_feat_path}")

fig_pix_path = os.path.join(RESULTS_DIR, f"fig_fora_pixel_{MODEL_TAG}.pdf")
fig, ax = plt.subplots(figsize=(6.5, 4.0))
_plot_curve(ax, "pixel",
            f"FORA pixel-space (oracle attacker)\n(model: {MODEL_TAG})")
plt.tight_layout(); plt.savefig(fig_pix_path); plt.close()
print(f"  wrote {fig_pix_path}")

# Qualitative figure: same FORA pixel decoder applied under each defense.
fig_qual_path = os.path.join(RESULTS_DIR, f"fig_fora_qualitative_{MODEL_TAG}.pdf")
N_QUAL = 4
qual_keys = [k for k in ("off", "fixed_0.5", "student") if k in all_results]
fig, axes = plt.subplots(len(qual_keys) + 1, N_QUAL,
                          figsize=(2.2 * N_QUAL, 2.2 * (len(qual_keys) + 1)))
if axes.ndim == 1: axes = axes.reshape(1, -1)
for j in range(N_QUAL):
    axes[0, j].imshow(pix_priv[j].permute(1, 2, 0).numpy())
    axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
    axes[0, j].set_title(f"sample {j+1}", fontsize=9)
    if j == 0:
        axes[0, j].set_ylabel("Original", fontsize=9)

# Reload the saved pixel-decoder state once
pix_dec = PixelDecoder().to(DEVICE)
pix_dec.load_state_dict({k: v.to(DEVICE) for k, v in _pix_state.items()})
pix_dec.eval()
for row, key in enumerate(qual_keys, start=1):
    sm = smashed_by_setting[key]["private"]["smashed"][:N_QUAL]
    sm_pad, mask = pad_batch(sm)
    with torch.no_grad():
        rec = pix_dec(sm_pad.to(DEVICE), mask.to(DEVICE)).cpu()
    for j in range(N_QUAL):
        axes[row, j].imshow(rec[j].permute(1, 2, 0).numpy().clip(0, 1))
        axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
        if j == 0:
            axes[row, j].set_ylabel(all_results[key]["label"], fontsize=8)
del pix_dec; free_cuda()

fig.suptitle(f"FORA qualitative (single decoder, varied defense)\n"
             f"model: {MODEL_TAG}", fontsize=10, y=1.02)
plt.tight_layout(); plt.savefig(fig_qual_path); plt.close()
print(f"  wrote {fig_qual_path}")


# =============================================================================
# 8) SAVE ARTIFACTS + UPLOAD
# =============================================================================
# Save the trained FORA pixel decoder so the qualitative figure can be
# regenerated later without re-running the whole attack pipeline.
fora_pix_path = os.path.join(RESULTS_DIR,
                              f"pix_decoder_fora_{MODEL_TAG}.pt")
torch.save(_pix_state, fora_pix_path)
print(f"[decoder] wrote {fora_pix_path}")

results_blob = {
    "attack":     "FORA",
    "model_tag":  MODEL_TAG,
    "dataset":    DATASET_KEY,
    "cut_layer":  CUT_LAYER,
    "n_public":   len(public_idx),
    "n_private":  len(private_idx),
    "config": {
        "epochs":          ATTACK_EPOCHS,
        "batch":           ATTACK_BATCH,
        "lr":              ATTACK_LR,
        "pixel_size":      PIXEL_SIZE,
        "rho_sweep":       RHO_SWEEP,
        "include_student": INCLUDE_STUDENT,
        "lora_r":          LORA_R,
        "lora_alpha":      LORA_ALPHA,
        "lora_dropout":    LORA_DROPOUT,
        "public_size":     PUBLIC_SIZE,
        "private_size":    PRIVATE_SIZE,
        "seed":            SEED,
    },
    "results":    all_results,
}
results_path = os.path.join(RESULTS_DIR, f"results_{RUN_TAG}.json")
with open(results_path, "w") as f:
    json.dump(results_blob, f, indent=2, default=str)
print(f"\n[results] wrote {results_path}")


def _safe_upload(local, remote):
    try:
        upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=HF_REPO, token=HF_TOKEN)
        print(f"  uploaded -> {HF_REPO}/{remote}")
    except Exception as e:
        print(f"  FAILED ({remote}): {e}")

if OFFLINE or not HF_TOKEN:
    print("\n[upload] skipping (offline or no token)")
else:
    print(f"\n[upload] pushing to {HF_REPO}")
    _safe_upload(results_path,  f"results_{RUN_TAG}.json")
    _safe_upload(fora_pix_path, f"decoders/pix_decoder_fora_{MODEL_TAG}.pt")
    _safe_upload(fig_feat_path, f"figures/fig_fora_feature_{MODEL_TAG}.pdf")
    _safe_upload(fig_pix_path,  f"figures/fig_fora_pixel_{MODEL_TAG}.pdf")
    _safe_upload(fig_qual_path, f"figures/fig_fora_qualitative_{MODEL_TAG}.pdf")


# =============================================================================
# 9) GC
# =============================================================================
print("\n[cleanup]")
del all_results, smashed_by_setting, clip_pub, clip_priv, pix_pub, pix_priv
del _pix_state
free_cuda()
print(f"[done] {gpu_status()}")
