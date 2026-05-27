"""
module_11b_fsha_original.py
===========================

Faithful FSHA (Feature-Stealing/Hijack) attack on SPLIT LEARNING for VQA-RAD.

Threat model (post-hoc simulation)
----------------------------------
True FSHA (Pasquini, Ateniese, Bernaschi 2021) trains the malicious server
ADVERSARIALLY DURING the client's training, poisoning gradients sent back
to the client so the client encoder converges to a shadow function `f_tilde`
that the attacker can invert. We attack already-trained LoRA checkpoints, so
we cannot poison training-time gradients. Instead we simulate the END STATE
of FSHA: train a shadow autoencoder `(f_tilde, f_tilde_inv)` on PUBLIC images,
adversarially align `f_tilde`'s output to the REAL client smashed activations
via a discriminator D, and use `f_tilde_inv` to invert PRIVATE smashed
activations into images. This is the standard post-hoc FSHA simulation used
in subsequent defense-evaluation literature.

Defense sweep (each is a separately-trained victim LoRA)
--------------------------------------------------------
  off, fixed rho=0.3, fixed rho=0.5, fixed rho=0.7, fixed rho=0.9, student

Outputs
-------
  results/results_fsha_orig_split_vqarad.json
  results/fig_fsha_orig_qualitative_split_vqarad.pdf  (4 rows x 7 cols)
  results/fsha_orig_images/*.png                       (28 individual cells)
  -> uploaded to HF Hub under FSHA_Updated/
"""

import os

# ============================================================================
# HARDCODED ENVIRONMENT (match the rest of the project)
# ============================================================================


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"
# ============================================================================

import gc
import json
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from huggingface_hub import upload_file

from qgtp_lib import (
    setup_hf, FrozenEncoders, load_student_from_hf, LLaVAWithQGTP,
    HF_REPO, HF_TOKEN, OFFLINE,
)
from module_10_attack_common import (
    DEVICE, CUT_LAYER, PIXEL_SIZE, PUBLIC_SIZE, PRIVATE_SIZE,
    load_lora_for_tag, public_private_split, load_dataset_with_attribute,
    SmashedExtractor, get_or_extract_smashed,
    PixelDecoder, pad_batch, get_pixel_targets_for_idx,
    free_cuda, gpu_status,
)

# =============================================================================
# CONFIG
# =============================================================================
DATASET_KEY = "vqarad"
SEED        = 42
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
CACHE_DIR   = os.path.join(RESULTS_DIR, "smashed_cache_fsha_orig")
IMG_DIR     = os.path.join(RESULTS_DIR, "fsha_orig_images")
for d in (RESULTS_DIR, CACHE_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

# LoRA hparams must match what the SL trainer used (module 8: r=32, alpha=64)
LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.05

# Each entry: (column_label, lora_tag, qgtp_mode, fixed_rho)
# The QGTP mode at attack-time MATCHES what that LoRA was trained with.
SETTINGS = [
    ("off",         "split_vqarad_off_cut16",            "off",     None),
    ("fixed_0.3",   "split_vqarad_fixed_cut16_rho0.3",   "fixed",   0.3),
    ("fixed_0.5",   "split_vqarad_fixed_cut16_rho0.5",   "fixed",   0.5),
    ("fixed_0.7",   "split_vqarad_fixed_cut16_rho0.7",   "fixed",   0.7),
    ("fixed_0.9",   "split_vqarad_fixed_cut16_rho0.9",   "fixed",   0.9),
    ("student",     "split_vqarad_student_cut16",        "student", None),
]
COL_PRETTY = {
    "off":       r"off ($\rho$=0)",
    "fixed_0.3": r"fixed ($\rho$=0.3)",
    "fixed_0.5": r"fixed ($\rho$=0.5)",
    "fixed_0.7": r"fixed ($\rho$=0.7)",
    "fixed_0.9": r"fixed ($\rho$=0.9)",
    "student":   r"student (learned $\rho$)",
}

# FSHA training schedule
FSHA_EPOCHS = 8
FSHA_BATCH  = 16
LR_F        = 1e-4    # shadow encoder
LR_FINV     = 1e-4    # shadow decoder
LR_D        = 2e-4    # discriminator
LAMBDA_ADV  = 0.2     # weight of adversarial term on f_tilde
LAMBDA_REC_REAL = 1.0 # weight of "real smashed -> image" anchor on f_inv

N_QUAL      = 4       # qualitative samples in the figure


# =============================================================================
# BANNER
# =============================================================================
print("=" * 72)
print("[FSHA-orig] active-hijack simulation on split-learning LoRAs (VQA-RAD)")
print(f"[cut={CUT_LAYER}] [pub={PUBLIC_SIZE}] [priv={PRIVATE_SIZE}] [{gpu_status()}]")
print("=" * 72)
setup_hf()
torch.manual_seed(SEED); np.random.seed(SEED)


# =============================================================================
# 1) DATA (shared across all 7 victim models)
# =============================================================================
print("\n[data] loading VQA-RAD")
samples, _attrs = load_dataset_with_attribute(
    DATASET_KEY, PUBLIC_SIZE + PRIVATE_SIZE + 200)
public_idx, private_idx = public_private_split(
    len(samples), PUBLIC_SIZE, PRIVATE_SIZE, seed=SEED)
print(f"[data] {len(samples)} samples -> public={len(public_idx)} "
      f"private={len(private_idx)}")

print("[data] precomputing pixel targets")
pix_pub  = get_pixel_targets_for_idx(samples, public_idx)
pix_priv = get_pixel_targets_for_idx(samples, private_idx)


# =============================================================================
# 2) FSHA ATTACK COMPONENTS
# =============================================================================
class ShadowEncoder(nn.Module):
    """f_tilde: image -> (T, D_lm) fake smashed sequence.

    Conv stem extracts a 24x24 spatial grid (matches CLIP-V patch grid),
    flattened to 576 tokens, then projected to D_lm=4096. We then trim to
    a target length T to mimic the kept-token count under the defense.
    """
    def __init__(self, out_tokens: int, out_dim: int = 4096, hidden: int = 256):
        super().__init__()
        self.out_tokens = out_tokens
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.GELU(),       # 56
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),     # 28
            nn.Conv2d(128, hidden, 3, 1, 1), nn.GELU(), # 28
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.stem(x)                          # (B, hidden, ~28, ~28)
        h = F.adaptive_avg_pool2d(h, (24, 24))    # (B, hidden, 24, 24)
        h = h.flatten(2).transpose(1, 2)          # (B, 576, hidden)
        h = self.proj(h)                          # (B, 576, D_lm)
        if h.size(1) > self.out_tokens:
            h = h[:, :self.out_tokens]
        return h


class Discriminator(nn.Module):
    """D: smashed sequence -> real/fake logit."""
    def __init__(self, in_dim: int = 4096, hidden: int = 192):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.attn    = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.norm    = nn.LayerNorm(hidden)
        self.head    = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, smashed_padded, key_padding_mask):
        kv = self.in_proj(smashed_padded)
        q  = kv.mean(dim=1, keepdim=True)
        q2, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                          need_weights=False)
        return self.head(self.norm(q + q2).squeeze(1)).squeeze(-1)


def fake_pad_batch(seq):
    """(B, T, D) fake smashed -> padded form + all-real mask (no padding)."""
    B, T, D = seq.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=seq.device)
    return seq, mask


# =============================================================================
# 3) FSHA TRAINING (per defense setting)
# =============================================================================
def fsha_train(real_pub_smashed, pix_pub_tensor, desc=""):
    """Joint adversarial training of (f_tilde, f_tilde_inv, D).

    Returns the trained f_tilde_inv (PixelDecoder) ready for inversion.
    """
    n = len(real_pub_smashed)
    # Use median real length so f_tilde produces sequences of comparable size
    real_lens = [t.size(0) for t in real_pub_smashed]
    target_T  = int(np.median(real_lens))

    f      = ShadowEncoder(out_tokens=target_T).to(DEVICE).train()
    f_inv  = PixelDecoder().to(DEVICE).train()
    disc   = Discriminator().to(DEVICE).train()

    opt_f     = torch.optim.AdamW(f.parameters(),     lr=LR_F)
    opt_finv  = torch.optim.AdamW(f_inv.parameters(), lr=LR_FINV)
    opt_d     = torch.optim.AdamW(disc.parameters(),  lr=LR_D)
    bce       = nn.BCEWithLogitsLoss()

    for ep in range(FSHA_EPOCHS):
        perm = np.random.permutation(n)
        pbar = tqdm(range(0, n, FSHA_BATCH),
                    desc=f"{desc} ep{ep+1}/{FSHA_EPOCHS}", leave=False)
        for s in pbar:
            ids   = perm[s:s + FSHA_BATCH].tolist()
            real  = [real_pub_smashed[i] for i in ids]
            real_pad, real_mask = pad_batch(real)
            real_pad  = real_pad.to(DEVICE)
            real_mask = real_mask.to(DEVICE)
            x_pub = pix_pub_tensor[ids].to(DEVICE)

            # ---- (a) Discriminator step: real=1, fake (from f_tilde)=0
            with torch.no_grad():
                fake_seq = f(x_pub)
            fake_pad, fake_mask = fake_pad_batch(fake_seq)
            d_real = disc(real_pad, real_mask)
            d_fake = disc(fake_pad, fake_mask)
            d_loss = bce(d_real, torch.ones_like(d_real)) + \
                     bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # ---- (b) Generator step: f_tilde fools D + reconstructs via f_inv
            fake_seq = f(x_pub)
            fake_pad, fake_mask = fake_pad_batch(fake_seq)
            d_fake = disc(fake_pad, fake_mask)
            adv_loss = bce(d_fake, torch.ones_like(d_fake))

            x_rec_shadow = f_inv(fake_pad, fake_mask)
            rec_shadow   = F.mse_loss(x_rec_shadow, x_pub)

            # ---- (c) Anchor: f_inv must also invert REAL smashed -> real img
            x_rec_real = f_inv(real_pad, real_mask)
            rec_real   = F.mse_loss(x_rec_real, x_pub)

            g_loss = rec_shadow + LAMBDA_REC_REAL * rec_real + LAMBDA_ADV * adv_loss
            opt_f.zero_grad(); opt_finv.zero_grad()
            g_loss.backward()
            opt_f.step(); opt_finv.step()

            pbar.set_postfix(d=f"{d_loss.item():.3f}",
                             g=f"{g_loss.item():.3f}",
                             adv=f"{adv_loss.item():.3f}")

    f_inv.eval()
    del f, disc, opt_f, opt_finv, opt_d
    free_cuda()
    return f_inv


@torch.no_grad()
def eval_inversion(f_inv, priv_smashed, pix_priv_tensor):
    """Per-sample PSNR / cosine / MSE on private set."""
    n = len(priv_smashed); psnrs, sims, mses = [], [], []
    for s in range(0, n, FSHA_BATCH):
        sm   = priv_smashed[s:s + FSHA_BATCH]
        pad, mask = pad_batch(sm)
        tgt  = pix_priv_tensor[s:s + FSHA_BATCH].to(DEVICE)
        pred = f_inv(pad.to(DEVICE), mask.to(DEVICE))
        m    = F.mse_loss(pred, tgt, reduction="none").mean(
                   dim=tuple(range(1, pred.ndim))).clamp(min=1e-12)
        psnrs.extend((-10.0 * torch.log10(m)).cpu().tolist())
        mses.extend(m.cpu().tolist())
        sims.extend(F.cosine_similarity(
            pred.flatten(1), tgt.flatten(1), dim=-1).cpu().tolist())
    return {
        "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
        "cos_mean":  float(np.mean(sims)),  "cos_std":  float(np.std(sims)),
        "mse_mean":  float(np.mean(mses)),
        "_per_sample_psnr": psnrs, "_per_sample_cos": sims,
    }


# =============================================================================
# 4) MAIN SWEEP: per LoRA, extract smashed -> train FSHA -> evaluate
# =============================================================================
print("\n[encoders] loading frozen CLIP/DINO + student (shared across runs)")
encoders = FrozenEncoders()
student  = load_student_from_hf()

all_results = {}
saved_decoders = {}   # for qualitative figure

# pick 4 fixed qualitative indices (within private set, deterministic)
rng = np.random.RandomState(SEED)
qual_idx = rng.choice(len(private_idx), size=N_QUAL, replace=False).tolist()

for col_key, lora_tag, qgtp_mode, fixed_rho in SETTINGS:
    print("\n" + "=" * 60)
    print(f"[FSHA] {col_key}  (LoRA: {lora_tag})")
    print("=" * 60)

    # --- load victim LLaVA + LoRA ------------------------------------------
    llava = LLaVAWithQGTP(lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                          lora_dropout=LORA_DROPOUT)
    load_lora_for_tag(llava, lora_tag, results_dir=RESULTS_DIR)
    llava.llava.eval()

    # --- extract smashed under the matching defense ------------------------
    extractor = SmashedExtractor(llava, encoders, student=student,
                                  cut_layer=CUT_LAYER)
    smashed = get_or_extract_smashed(
        extractor, samples, public_idx, private_idx,
        CACHE_DIR, lora_tag, qgtp_mode, fixed_rho)
    extractor.close()

    # release victim before FSHA training (frees ~14GB)
    del llava, extractor; free_cuda()

    sm_pub  = smashed["public"]["smashed"]
    sm_priv = smashed["private"]["smashed"]
    rhos    = smashed["private"]["rhos"]
    n_kept  = smashed["private"]["n_kept"]

    # --- FSHA adversarial training ----------------------------------------
    f_inv = fsha_train(sm_pub, pix_pub, desc=col_key)

    # --- inversion eval on private ----------------------------------------
    metrics = eval_inversion(f_inv, sm_priv, pix_priv)

    # cache reconstructions for the 4 qualitative samples
    qual_sm = [sm_priv[i] for i in qual_idx]
    qual_pad, qual_mask = pad_batch(qual_sm)
    with torch.no_grad():
        qual_rec = f_inv(qual_pad.to(DEVICE), qual_mask.to(DEVICE)).cpu()

    qual_psnr = [metrics["_per_sample_psnr"][i] for i in qual_idx]
    qual_cos  = [metrics["_per_sample_cos"][i]  for i in qual_idx]

    saved_decoders[col_key] = {
        "rec":      qual_rec,           # (N_QUAL, 3, 112, 112)
        "psnr":     qual_psnr,
        "cos":      qual_cos,
    }

    all_results[col_key] = {
        "label":         COL_PRETTY[col_key],
        "lora_tag":      lora_tag,
        "qgtp_mode":     qgtp_mode,
        "fixed_rho":     fixed_rho,
        "rho_mean":      float(np.mean(rhos)),
        "rho_std":       float(np.std(rhos)),
        "n_kept_mean":   float(np.mean(n_kept)),
        "psnr_mean":     metrics["psnr_mean"],
        "psnr_std":      metrics["psnr_std"],
        "cos_mean":      metrics["cos_mean"],
        "cos_std":       metrics["cos_std"],
        "mse_mean":      metrics["mse_mean"],
    }
    print(f"   rho_mean={np.mean(rhos):.3f}  n_kept={np.mean(n_kept):.0f}/576")
    print(f"   PSNR={metrics['psnr_mean']:.2f} +/- {metrics['psnr_std']:.2f} dB   "
          f"cos={metrics['cos_mean']:.3f} +/- {metrics['cos_std']:.3f}")

    del f_inv, smashed, sm_pub, sm_priv; free_cuda()

# free shared models
del encoders, student; free_cuda()
print(f"\n[free] after sweep: {gpu_status()}")


# =============================================================================
# 5) SAVE 28 INDIVIDUAL IMAGES (4 originals + 4 x 6 reconstructions)
# =============================================================================
print("\n[images] saving 28 individual cells")
def _save_cell(path, img_tensor, title=None):
    """img_tensor: (3, H, W) in [0,1]."""
    arr = img_tensor.permute(1, 2, 0).numpy().clip(0, 1)
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.imshow(arr); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# originals
for j, qi in enumerate(qual_idx):
    p = os.path.join(IMG_DIR, f"original_sample{j+1}.png")
    _save_cell(p, pix_priv[qi], title=f"Original (sample {j+1})")

# reconstructions
for col_key, _, _, _ in SETTINGS:
    rec   = saved_decoders[col_key]["rec"]
    psnrs = saved_decoders[col_key]["psnr"]
    coss  = saved_decoders[col_key]["cos"]
    for j in range(N_QUAL):
        p = os.path.join(IMG_DIR, f"{col_key}_sample{j+1}.png")
        title = (f"{COL_PRETTY[col_key]}\nPSNR={psnrs[j]:.2f} dB  "
                 f"cos={coss[j]:.3f}")
        _save_cell(p, rec[j], title=title)
print(f"  wrote {4 + 6 * N_QUAL} files to {IMG_DIR}")


# =============================================================================
# 6) COMPOSITE FIGURE (4 rows x 7 cols, labelled for paper)
# =============================================================================
print("[figure] composing 4 x 7 qualitative grid")
COLS = ["original"] + [k for k, *_ in SETTINGS]   # 7 columns
COL_TITLES = ["Original"] + [COL_PRETTY[k] for k, *_ in SETTINGS]

fig, axes = plt.subplots(N_QUAL, 7, figsize=(2.3 * 7, 2.6 * N_QUAL))
for j in range(N_QUAL):
    for c, col in enumerate(COLS):
        ax = axes[j, c]
        if col == "original":
            ax.imshow(pix_priv[qual_idx[j]].permute(1, 2, 0).numpy().clip(0, 1))
            sub = ""
        else:
            ax.imshow(saved_decoders[col]["rec"][j].permute(1, 2, 0).numpy().clip(0, 1))
            sub = (f"PSNR={saved_decoders[col]['psnr'][j]:.2f} dB\n"
                   f"cos={saved_decoders[col]['cos'][j]:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_title(COL_TITLES[c], fontsize=10)
        if c == 0:
            ax.set_ylabel(f"sample {j+1}", fontsize=10)
        if sub:
            # put per-image metrics under the panel
            ax.set_xlabel(sub, fontsize=8)

fig.suptitle(
    "FSHA (active-hijack simulation) on Split-Learning -- VQA-RAD\n"
    "Reconstructions from private smashed activations at LM layer 16",
    fontsize=11, y=1.005)
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "fig_fsha_orig_qualitative_split_vqarad.pdf")
plt.savefig(fig_path, bbox_inches="tight"); plt.close(fig)
print(f"  wrote {fig_path}")


# =============================================================================
# 7) SAVE JSON SUMMARY
# =============================================================================
print("[results] writing JSON")
results_blob = {
    "attack":       "FSHA (Feature-Stealing/Hijack Attack, active simulation)",
    "threat_model": (
        "Original FSHA (Pasquini et al. 2021) poisons gradients sent back to "
        "the client during the client's training so the client encoder "
        "converges to a shadow function f_tilde the attacker can invert. "
        "Here, victims (LoRAs) are already trained and frozen, so we simulate "
        "the END STATE of FSHA: a shadow autoencoder (f_tilde, f_tilde_inv) "
        "trained on public images, distributionally aligned to real client "
        "smashed activations via a discriminator D. We then invert PRIVATE "
        "smashed with f_tilde_inv. Gradient-poisoning during victim training "
        "is OUT OF SCOPE for this experiment."
    ),
    "dataset":      DATASET_KEY,
    "setting":      "split_learning",
    "cut_layer":    CUT_LAYER,
    "n_public":     len(public_idx),
    "n_private":    len(private_idx),
    "qual_indices": qual_idx,
    "config": {
        "fsha_epochs":     FSHA_EPOCHS,
        "fsha_batch":      FSHA_BATCH,
        "lr_f":            LR_F,
        "lr_finv":         LR_FINV,
        "lr_d":            LR_D,
        "lambda_adv":      LAMBDA_ADV,
        "lambda_rec_real": LAMBDA_REC_REAL,
        "pixel_size":      PIXEL_SIZE,
        "lora_r":          LORA_R,
        "lora_alpha":      LORA_ALPHA,
        "lora_dropout":    LORA_DROPOUT,
        "seed":            SEED,
    },
    "results":     all_results,
    "qualitative": {
        col_key: {
            "psnr_per_sample": saved_decoders[col_key]["psnr"],
            "cos_per_sample":  saved_decoders[col_key]["cos"],
        } for col_key, *_ in SETTINGS
    },
}
results_path = os.path.join(RESULTS_DIR, "results_fsha_orig_split_vqarad.json")
with open(results_path, "w") as f:
    json.dump(results_blob, f, indent=2, default=str)
print(f"  wrote {results_path}")


# =============================================================================
# 8) UPLOAD TO HF UNDER FSHA_Updated/
# =============================================================================
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
    print(f"\n[upload] -> {HF_REPO}/FSHA_Updated/")
    _safe_upload(results_path, "FSHA_Updated/results_fsha_orig_split_vqarad.json")
    _safe_upload(fig_path,     "FSHA_Updated/fig_fsha_orig_qualitative_split_vqarad.pdf")
    # upload individual images
    for fname in sorted(os.listdir(IMG_DIR)):
        _safe_upload(os.path.join(IMG_DIR, fname),
                     f"FSHA_Updated/images/{fname}")

print(f"\n[done] {gpu_status()}")
