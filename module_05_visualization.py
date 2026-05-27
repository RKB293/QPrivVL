"""
module_05_visualization.py
==========================

Generates publication-quality figures for the QGTP pipeline.

Figures produced:
  fig01_pipeline_<dataset>_v{k}.pdf
      Per-sample decomposition: image | sensitivity | utility | combined |
      keep-mask. K variants per dataset.

  fig02_teacher_vs_student_<dataset>_v{k}.pdf
      Teacher vs. student keep-mask comparison with IoU and agreement badges.

  fig03_rho_agreement.pdf
      Global teacher-vs-student rho scatter + per-dataset MAE bar chart.

  fig05_kept_token_distribution.pdf
      Per-dataset histogram of kept-token counts.

  fig06_07_rho_and_kept_vs_sens.pdf
      Per-dataset rho scatter (top row) and kept-tokens vs. max-sensitivity
      scatter (bottom row).

Usage
-----
    python module_05_visualization.py
"""

import os
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

from huggingface_hub import login, hf_hub_download, upload_file

# =============================================================================
# CONFIG
# =============================================================================
HF_USER  = os.environ.get("HF_USER",  "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO  = os.environ.get("HF_REPO",  f"{HF_USER}/QprivVL")

DATASETS_FOR_PER_SAMPLE_FIGS = ["vqarad", "vqav2", "gqa"]
ALL_DATASETS    = ["vqav2", "gqa", "okvqa", "slake", "vqarad", "pathvqa"]
N_SAMPLES_PER_FIG = 4
N_VARIANTS        = 3
N_SCATTER_PER_DS  = 200
CLIP_GRID_SIDE    = 24
DINO_GRID_SIDE    = 16
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = os.environ.get("RESULTS_DIR", "./results/viz")
os.makedirs(OUT_DIR, exist_ok=True)

CMAP_SENS     = "Reds"
CMAP_UTIL     = "Blues"
CMAP_COMBINED = "RdBu_r"
CMAP_MASK     = "Greens"
IMAGE_ALPHA   = 0.55
OVERLAY_ALPHA = 0.78
PAGE_WIDTH_DOUBLE = 7.0

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "patch.linewidth": 0.5,
    "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.dpi": 150,
})

DS_COLORS = {
    "vqav2": "#1f77b4", "gqa": "#ff7f0e", "okvqa": "#2ca02c",
    "slake": "#d62728", "vqarad": "#9467bd", "pathvqa": "#8c564b",
}
DS_PRETTY = {
    "vqav2": "VQAv2", "gqa": "GQA", "okvqa": "OK-VQA",
    "slake": "SLAKE", "vqarad": "VQA-RAD", "pathvqa": "PathVQA",
}

assert HF_TOKEN and HF_USER, "Set HF_TOKEN and HF_USER before running."

# =============================================================================
# MODEL CLASSES (must match modules 3 and 4 exactly)
# =============================================================================

class SensitivityHead(nn.Module):
    def __init__(self, in_dim=384, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patches):
        return torch.sigmoid(self.net(patches).squeeze(-1))


class ThresholdPredictor(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class QGTPScorer(nn.Module):
    def __init__(self, vis_dim, txt_dim, proj_dim=256):
        super().__init__()
        self.Wq = nn.Linear(txt_dim, proj_dim, bias=False)
        self.Wk = nn.Linear(vis_dim, proj_dim, bias=False)
        self.scale = proj_dim ** 0.5

    def forward(self, visual, text):
        return torch.einsum("bd,bnd->bn", self.Wq(text), self.Wk(visual)) / self.scale


class Student(nn.Module):
    def __init__(self, vis_dim, txt_dim, dino_dim, teacher_cfg, sens_cfg,
                 alpha=1.0, beta=1.0):
        super().__init__()
        self.sens_head = SensitivityHead(**sens_cfg)
        self.predictor = ThresholdPredictor(**teacher_cfg)
        self.scorer    = QGTPScorer(vis_dim, txt_dim)
        self.gamma     = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("_alpha", torch.tensor(float(alpha)))
        self.register_buffer("_beta",  torch.tensor(float(beta)))

    @staticmethod
    def _align_spatial(sens_dino, clip_side=24, dino_side=16):
        B = sens_dino.size(0)
        up = F.interpolate(sens_dino.view(B, 1, dino_side, dino_side),
                           size=(clip_side, clip_side), mode="bilinear", align_corners=False)
        return up.view(B, clip_side * clip_side)

    @staticmethod
    def _sens_summary(sens_dino):
        return torch.cat([
            sens_dino.mean(-1, keepdim=True),
            sens_dino.max(-1, keepdim=True).values,
            (sens_dino > 0.5).float().mean(-1, keepdim=True),
        ], dim=-1)

    def forward(self, clip_patches, clip_pooled, clip_text, dino_patches):
        sens_dino    = self.sens_head(dino_patches)
        sens_aligned = self._align_spatial(sens_dino)
        util_norm    = torch.sigmoid(self.scorer(clip_patches, clip_text))
        combined     = self._alpha * util_norm - self._beta * sens_aligned
        sens_sum     = self._sens_summary(sens_dino)
        rho_hat      = self.predictor(torch.cat([clip_pooled, clip_text, sens_sum], dim=-1))
        mask_prob    = torch.sigmoid(self.gamma * combined)
        return {"rho": rho_hat, "mask": mask_prob, "util": util_norm,
                "sens": sens_aligned, "sens_dino": sens_dino, "combined": combined}


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
print("[load] downloading artifacts")
login(token=HF_TOKEN)

sens_art    = torch.load(hf_hub_download(HF_REPO, "sensitivity_head.pt",  token=HF_TOKEN), map_location="cpu")
teacher_art = torch.load(hf_hub_download(HF_REPO, "teacher_predictor.pt", token=HF_TOKEN), map_location="cpu")
student_art = torch.load(hf_hub_download(HF_REPO, "student_model.pt",     token=HF_TOKEN), map_location="cpu")

sens_teacher = SensitivityHead(**sens_art["config"]).eval()
sens_teacher.load_state_dict(sens_art["state_dict"])

teacher_pred = ThresholdPredictor(**teacher_art["config"]).eval()
teacher_pred.load_state_dict(teacher_art["state_dict"])

student = Student(**student_art["config"]).eval()
student.load_state_dict(student_art["state_dict"])

teacher_scorer = QGTPScorer(
    vis_dim=student_art["config"]["vis_dim"],
    txt_dim=student_art["config"]["txt_dim"],
).eval()

# =============================================================================
# HELPERS
# =============================================================================

def percentile_norm(values, low=5.0, high=95.0):
    arr = values.detach().cpu().numpy().ravel()
    lo, hi = float(np.percentile(arr, low)), float(np.percentile(arr, high))
    return (0.0, 1.0) if hi - lo < 1e-6 else (lo, hi)


def symmetric_norm(values, percentile=95.0):
    v = float(np.percentile(np.abs(values.detach().cpu().numpy().ravel()), percentile))
    v = max(v, 1e-6)
    return -v, v


def heatmap_image(values, side, image_size):
    grid = values.view(side, side).cpu().numpy().astype(np.float32)
    pil  = Image.fromarray(grid, mode="F").resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(pil)


def overlay_on_image(ax, img_arr, heat, cmap, norm):
    ax.imshow(img_arr, alpha=IMAGE_ALPHA)
    ax.imshow(heat, alpha=OVERLAY_ALPHA, cmap=cmap, norm=norm, interpolation="bilinear")


def teacher_forward(cp, cpo, ct, dp):
    sens_dino    = sens_teacher(dp)
    sens_aligned = Student._align_spatial(sens_dino)
    util_norm    = torch.sigmoid(teacher_scorer(cp, ct))
    alpha = student._alpha.item(); beta = student._beta.item()
    combined    = alpha * util_norm - beta * sens_aligned
    mask_target = torch.sigmoid(3.0 * combined)
    sens_sum    = Student._sens_summary(sens_dino)
    rho_target  = teacher_pred(torch.cat([cpo, ct, sens_sum], dim=-1))
    return {"rho": rho_target, "mask": mask_target, "util": util_norm,
            "sens": sens_aligned, "sens_dino": sens_dino, "combined": combined}


def reload_image(idx, dataset_key):
    from datasets import load_dataset
    cache = reload_image._cache
    if dataset_key not in cache:
        loaders = {
            "vqarad":   lambda: load_dataset("flaviagiammarino/vqa-rad",   split="train"),
            "pathvqa":  lambda: load_dataset("flaviagiammarino/path-vqa",  split="train"),
            "slake":    lambda: load_dataset("mdwiratathya/SLAKE-vqa-english", split="train"),
            "vqav2":    lambda: load_dataset("lmms-lab/VQAv2",            split="validation"),
            "okvqa":    lambda: load_dataset("lmms-lab/OK-VQA",           split="validation"),
        }
        if dataset_key in loaders:
            cache[dataset_key] = loaders[dataset_key]()
        else:
            return None
    ds = cache[dataset_key]
    return ds[idx]["image"].convert("RGB") if idx < len(ds) else None

reload_image._cache = {}


def shorten(s, n):
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + "..."


def save_fig(fig, base_path):
    fig.savefig(base_path + ".pdf")
    fig.savefig(base_path + ".png")
    plt.close(fig)


def pick_diverse_samples(feat_artifact, n_samples, rng_seed):
    N = feat_artifact["clip_patches"].size(0)
    max_sens = []
    for s in range(0, N, 64):
        chunk = feat_artifact["dino_patches"][s:s+64].float()
        with torch.no_grad():
            max_sens.append(sens_teacher(chunk).max(dim=-1).values)
    max_sens = torch.cat(max_sens).numpy()
    order    = np.argsort(max_sens)
    rng      = np.random.RandomState(rng_seed)
    picks    = [int(rng.choice(chunk)) for chunk in np.array_split(order, n_samples)]
    return picks, max_sens


# =============================================================================
# FIG 01: pipeline decomposition
# =============================================================================
def render_fig01(feat, ds_key, variant_idx, sample_idxs, max_sens, out_dir):
    n   = len(sample_idxs)
    cp  = feat["clip_patches"][sample_idxs].float()
    cpo = feat["clip_pooled" ][sample_idxs].float()
    ct  = feat["clip_text"   ][sample_idxs].float()
    dp  = feat["dino_patches"][sample_idxs].float()
    qs  = [feat["questions"][i] for i in sample_idxs]
    ans = [feat["answers"  ][i] for i in sample_idxs]
    PANEL_PIX = 280

    with torch.no_grad():
        t_out = teacher_forward(cp, cpo, ct, dp)
        s_out = student(cp, cpo, ct, dp)

    sample_images = [reload_image(i, ds_key) for i in sample_idxs]
    panel_titles  = ["Image", r"Sensitivity $s$", r"Utility $u$",
                     r"Combined $\alpha u-\beta s$", r"Keep-mask $m$"]

    fig = plt.figure(figsize=(PAGE_WIDTH_DOUBLE, 0.95 * PAGE_WIDTH_DOUBLE / 5 * n + 0.7))
    gs  = GridSpec(n + 1, 6, figure=fig, wspace=0.04, hspace=0.20,
                   width_ratios=[1, 1, 1, 1, 1, 1.55], height_ratios=[1] * n + [0.10])

    for r in range(n):
        img = (sample_images[r] or Image.new("RGB", (PANEL_PIX, PANEL_PIX), (220, 220, 220)))
        img = img.resize((PANEL_PIX, PANEL_PIX), Image.BILINEAR)
        img_arr = np.asarray(img)

        sens_raw = s_out["sens"][r]; util_raw = s_out["util"][r]
        comb_raw = s_out["combined"][r]; mask_raw = s_out["mask"][r]
        signals = [
            (None, None, None),
            (heatmap_image(sens_raw, CLIP_GRID_SIDE, PANEL_PIX), CMAP_SENS,
             Normalize(*percentile_norm(sens_raw))),
            (heatmap_image(util_raw, CLIP_GRID_SIDE, PANEL_PIX), CMAP_UTIL,
             Normalize(*percentile_norm(util_raw))),
            (heatmap_image(comb_raw, CLIP_GRID_SIDE, PANEL_PIX), CMAP_COMBINED,
             Normalize(*symmetric_norm(comb_raw))),
            (heatmap_image(mask_raw, CLIP_GRID_SIDE, PANEL_PIX), CMAP_MASK,
             Normalize(0.0, 1.0)),
        ]
        for c in range(5):
            ax = fig.add_subplot(gs[r, c])
            heat, cmap, norm = signals[c]
            if heat is None:
                ax.imshow(img_arr)
            else:
                overlay_on_image(ax, img_arr, heat, cmap, norm)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(panel_titles[c], fontsize=8, pad=4)

        ax_cap = fig.add_subplot(gs[r, 5])
        ax_cap.axis("off")
        n_kept = int((mask_raw > 0.5).sum())
        caption = (
            r"$\mathrm{idx}\,$" + f"{sample_idxs[r]}\n"
            f"Q: {shorten(qs[r], 60)}\nA: {shorten(ans[r], 35)}\n"
            r"$\rho_T=$" + f"{t_out['rho'][r].item():.2f},  "
            r"$\rho_S=$" + f"{s_out['rho'][r].item():.2f}\n"
            f"kept = {n_kept}/576\n"
            r"$\max_p s_p=$" + f"{t_out['sens_dino'][r].max().item():.2f}"
        )
        ax_cap.text(0.0, 0.94, caption, va="top", ha="left", family="serif", fontsize=7.0,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5",
                              edgecolor="#cccccc", linewidth=0.4))

    cbar_specs = [
        (1, CMAP_SENS,     Normalize(0, 1),  "sensitivity"),
        (2, CMAP_UTIL,     Normalize(0, 1),  "utility"),
        (3, CMAP_COMBINED, Normalize(-1, 1), "combined"),
        (4, CMAP_MASK,     Normalize(0, 1),  "keep prob."),
    ]
    for col, cmap, norm, label in cbar_specs:
        cax = fig.add_subplot(gs[n, col])
        sm  = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=7, labelpad=2)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)

    fig.suptitle(f"Pipeline on {DS_PRETTY.get(ds_key, ds_key)} (variant {variant_idx+1})",
                 fontsize=9, y=1.00)
    base = os.path.join(out_dir, f"fig01_pipeline_{ds_key}_v{variant_idx+1}")
    save_fig(fig, base)
    return base + ".pdf"


# =============================================================================
# FIG 02: teacher vs student
# =============================================================================
def render_fig02(feat, ds_key, variant_idx, sample_idxs, out_dir):
    n   = len(sample_idxs)
    cp  = feat["clip_patches"][sample_idxs].float()
    cpo = feat["clip_pooled" ][sample_idxs].float()
    ct  = feat["clip_text"   ][sample_idxs].float()
    dp  = feat["dino_patches"][sample_idxs].float()
    PANEL_PIX = 260

    with torch.no_grad():
        t_out = teacher_forward(cp, cpo, ct, dp)
        s_out = student(cp, cpo, ct, dp)

    sample_images = [reload_image(i, ds_key) for i in sample_idxs]
    fig = plt.figure(figsize=(PAGE_WIDTH_DOUBLE, 0.95 * PAGE_WIDTH_DOUBLE / 5 * n + 0.5))
    gs  = GridSpec(n + 1, 5, figure=fig, wspace=0.04, hspace=0.20,
                   height_ratios=[1] * n + [0.10])

    for r in range(n):
        img = (sample_images[r] or Image.new("RGB", (PANEL_PIX, PANEL_PIX), (220, 220, 220)))
        img = img.resize((PANEL_PIX, PANEL_PIX), Image.BILINEAR)
        img_arr = np.asarray(img)
        t_mask = t_out["mask"][r]; s_mask = s_out["mask"][r]
        t_bin  = (t_mask > 0.5).float(); s_bin = (s_mask > 0.5).float()
        diff   = (s_mask - t_mask).abs()
        inter  = ((t_bin == 1) & (s_bin == 1)).float().sum()
        union  = ((t_bin == 1) | (s_bin == 1)).float().sum().clamp(min=1)
        iou    = (inter / union).item()
        agree  = (t_bin == s_bin).float().mean().item()

        cells = [
            ("Image", None, None, None, r"$\mathrm{idx}\," + f"{sample_idxs[r]}$"),
            ("Teacher (soft)", heatmap_image(t_mask, CLIP_GRID_SIDE, PANEL_PIX),
             CMAP_MASK, Normalize(0, 1), r"$\rho_T=$" + f"{t_out['rho'][r].item():.2f}"),
            ("Student (soft)", heatmap_image(s_mask, CLIP_GRID_SIDE, PANEL_PIX),
             CMAP_MASK, Normalize(0, 1), r"$\rho_S=$" + f"{s_out['rho'][r].item():.2f}"),
            ("Student (binary)", heatmap_image(s_bin, CLIP_GRID_SIDE, PANEL_PIX),
             CMAP_MASK, Normalize(0, 1), f"keep={int(s_bin.sum())}/576"),
            (r"$|m_T-m_S|$", heatmap_image(diff, CLIP_GRID_SIDE, PANEL_PIX),
             "Purples", Normalize(*percentile_norm(diff)),
             f"IoU={iou:.2f} agr={agree:.2f}"),
        ]
        for c, (title, hm, cmap, norm, badge) in enumerate(cells):
            ax = fig.add_subplot(gs[r, c])
            if hm is None:
                ax.imshow(img_arr)
            else:
                overlay_on_image(ax, img_arr, hm, cmap, norm)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=8, pad=4)
            ax.text(0.02, 0.98, badge, transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.8, color="white",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="black",
                              alpha=0.62, edgecolor="none"))

    for ax, sm_spec in [
        (fig.add_subplot(gs[n, 1]), (CMAP_MASK,  Normalize(0, 1), "keep probability")),
        (fig.add_subplot(gs[n, 4]), ("Purples",  Normalize(0, 1), r"|teacher$-$student|")),
    ]:
        sm = ScalarMappable(norm=sm_spec[1], cmap=sm_spec[0]); sm.set_array([])
        cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
        cb.set_label(sm_spec[2], fontsize=7, labelpad=2)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4)

    fig.suptitle(f"Teacher vs. student keep-masks on {DS_PRETTY.get(ds_key, ds_key)} "
                 f"(variant {variant_idx+1})", fontsize=9, y=1.00)
    base = os.path.join(out_dir, f"fig02_teacher_vs_student_{ds_key}_v{variant_idx+1}")
    save_fig(fig, base)
    return base + ".pdf"


# =============================================================================
# AGGREGATE METRICS
# =============================================================================
print("\n[agg] computing per-dataset aggregates")
per_ds_rho, per_ds_mask, per_ds_sens = {}, {}, {}

for ds_key in ALL_DATASETS:
    try:
        f = torch.load(hf_hub_download(HF_REPO, f"features_{ds_key}.pt", token=HF_TOKEN),
                       map_location="cpu")
    except Exception as e:
        print(f"  skip {ds_key}: {e}"); continue

    n    = min(N_SCATTER_PER_DS, f["clip_patches"].size(0))
    perm = np.random.RandomState(0).permutation(f["clip_patches"].size(0))[:n]
    cp_  = f["clip_patches"][perm].float(); cpo_ = f["clip_pooled"][perm].float()
    ct_  = f["clip_text"  ][perm].float(); dp_  = f["dino_patches"][perm].float()

    rho_t_c, rho_s_c, mask_c, sens_c = [], [], [], []
    for s in range(0, n, 64):
        with torch.no_grad():
            t    = teacher_forward(cp_[s:s+64], cpo_[s:s+64], ct_[s:s+64], dp_[s:s+64])
            sout = student(cp_[s:s+64], cpo_[s:s+64], ct_[s:s+64], dp_[s:s+64])
        rho_t_c.append(t["rho"]); rho_s_c.append(sout["rho"])
        mask_c.append((sout["mask"] > 0.5).sum(dim=-1))
        sens_c.append(t["sens_dino"].max(dim=-1).values)
    per_ds_rho[ds_key]  = (torch.cat(rho_t_c).numpy(), torch.cat(rho_s_c).numpy())
    per_ds_mask[ds_key] = torch.cat(mask_c).numpy()
    per_ds_sens[ds_key] = torch.cat(sens_c).numpy()
    print(f"  {ds_key}: {n} samples")

ds_keys    = list(per_ds_rho.keys())
ds_maes    = [float(np.mean(np.abs(per_ds_rho[k][0] - per_ds_rho[k][1]))) for k in ds_keys]
all_rho_t  = np.concatenate([v[0] for v in per_ds_rho.values()])
all_rho_s  = np.concatenate([v[1] for v in per_ds_rho.values()])
mae_global  = float(np.mean(np.abs(all_rho_t - all_rho_s)))
corr_global = float(np.corrcoef(all_rho_t, all_rho_s)[0, 1])

# =============================================================================
# FIG 03: rho agreement
# =============================================================================
print("\n[fig03] rho agreement")
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_DOUBLE, 2.5),
                          gridspec_kw={"width_ratios": [1.0, 1.0]})
ax = axes[0]
for ds_key, (rt, rs) in per_ds_rho.items():
    ax.scatter(rt, rs, s=7, alpha=0.6, color=DS_COLORS.get(ds_key, "#444"),
               edgecolor="none", label=DS_PRETTY.get(ds_key, ds_key))
ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.7, label=r"$y=x$")
ax.set_xlabel(r"Teacher $\rho_T$"); ax.set_ylabel(r"Student $\rho_S$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title(f"(a) MAE = {mae_global:.3f}, Pearson $r$ = {corr_global:.3f}", fontsize=8)
ax.legend(loc="upper left", framealpha=0.92, ncol=2, fontsize=6,
          handletextpad=0.2, columnspacing=0.6)
ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

ax = axes[1]
xs = np.arange(len(ds_keys))
ax.bar(xs, ds_maes, color=[DS_COLORS.get(k, "#444") for k in ds_keys],
       edgecolor="black", linewidth=0.4)
for x, m in zip(xs, ds_maes):
    ax.text(x, m + 0.001, f"{m:.3f}", ha="center", va="bottom", fontsize=6)
ax.axhline(mae_global, color="black", linestyle="--", linewidth=0.6,
           label=f"global MAE $=$ {mae_global:.3f}")
ax.set_xticks(xs)
ax.set_xticklabels([DS_PRETTY.get(k, k) for k in ds_keys], rotation=20, ha="right")
ax.set_ylabel(r"MAE on $\rho$")
ax.set_title("(b) Per-dataset distillation gap", fontsize=8)
ax.legend(loc="upper left", fontsize=6, framealpha=0.92)
ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
fig.tight_layout()
fig03_path = os.path.join(OUT_DIR, "fig03_rho_agreement")
save_fig(fig, fig03_path)
print(f"  saved {fig03_path}.pdf")

# =============================================================================
# FIG 05: kept-token distribution
# =============================================================================
print("\n[fig05] kept-token distribution")
fig, ax = plt.subplots(figsize=(PAGE_WIDTH_DOUBLE, 2.3))
bin_edges = np.linspace(0, 576, 31)
for ds_key, kept in per_ds_mask.items():
    ax.hist(kept, bins=bin_edges, alpha=0.6, color=DS_COLORS.get(ds_key, "#444"),
            edgecolor="black", linewidth=0.3,
            label=f"{DS_PRETTY.get(ds_key, ds_key)} ({kept.mean():.0f})")
ax.set_xlabel(r"$\#$ kept tokens (mask $> 0.5$)"); ax.set_ylabel(r"$\#$ samples")
ax.set_title("Per-dataset pruning aggressiveness", fontsize=8)
ax.legend(loc="upper center", ncol=4, fontsize=6, framealpha=0.92)
ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5); ax.set_xlim(0, 576)
fig.tight_layout()
fig05_path = os.path.join(OUT_DIR, "fig05_kept_token_distribution")
save_fig(fig, fig05_path)

# =============================================================================
# FIG 06/07: per-dataset rho + kept-vs-sensitivity
# =============================================================================
print("\n[fig06_07] per-dataset rho and kept-vs-sensitivity")
n_ds = len(ds_keys)
fig, axes = plt.subplots(2, n_ds, figsize=(1.45 * n_ds + 0.5, 4.6),
                          squeeze=False, sharey="row")
for col, ds_key in enumerate(ds_keys):
    rt, rs = per_ds_rho[ds_key]
    mae = float(np.mean(np.abs(rt - rs))); r = float(np.corrcoef(rt, rs)[0, 1])
    ax = axes[0, col]
    ax.scatter(rt, rs, s=7, alpha=0.6, color=DS_COLORS.get(ds_key, "#444"), edgecolor="none")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{DS_PRETTY.get(ds_key, ds_key)}\nMAE={mae:.3f}, r={r:.2f}", fontsize=7.5)
    if col == 0:
        ax.set_ylabel(r"Student $\rho_S$")
    ax.set_xlabel(r"Teacher $\rho_T$", fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.tick_params(labelsize=6.5)

    sens = per_ds_sens[ds_key]; kept = per_ds_mask[ds_key]
    rcoef = float(np.corrcoef(sens, kept)[0, 1]) if len(sens) > 1 and sens.std() > 0 else float("nan")
    ax = axes[1, col]
    ax.scatter(sens, kept, s=7, alpha=0.6, color=DS_COLORS.get(ds_key, "#444"), edgecolor="none")
    ax.set_xlim(0, 1); ax.set_ylim(-10, 590)
    ax.set_title(f"r={rcoef:.2f}", fontsize=7.5)
    if col == 0:
        ax.set_ylabel(r"$\#$ kept tokens")
    ax.set_xlabel(r"$\max_p s_p$", fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.tick_params(labelsize=6.5)

fig.suptitle("Per-dataset distillation fidelity (top) and privacy-driven pruning (bottom)",
             fontsize=9, y=1.01)
fig.tight_layout()
fig0607_path = os.path.join(OUT_DIR, "fig06_07_rho_and_kept_vs_sens")
save_fig(fig, fig0607_path)

# =============================================================================
# PER-SAMPLE / PER-VARIANT FIGURES
# =============================================================================
print("\n[per-sample] generating per-dataset, per-variant figures")
per_sample_results = {}
for ds_key in DATASETS_FOR_PER_SAMPLE_FIGS:
    try:
        feat = torch.load(hf_hub_download(HF_REPO, f"features_{ds_key}.pt", token=HF_TOKEN),
                          map_location="cpu")
    except Exception as e:
        print(f"  skip {ds_key}: {e}"); continue

    ds_dir = os.path.join(OUT_DIR, ds_key)
    os.makedirs(ds_dir, exist_ok=True)
    per_sample_results[ds_key] = {"fig01": [], "fig02": []}

    for v in range(N_VARIANTS):
        sample_idxs, max_sens = pick_diverse_samples(feat, N_SAMPLES_PER_FIG, 1234 + 7 * v)
        print(f"  [{ds_key} v{v+1}] idxs = {sample_idxs}")
        try:
            per_sample_results[ds_key]["fig01"].append(
                render_fig01(feat, ds_key, v, sample_idxs, max_sens, ds_dir))
            per_sample_results[ds_key]["fig02"].append(
                render_fig02(feat, ds_key, v, sample_idxs, ds_dir))
        except Exception as e:
            print(f"    WARN variant {v+1} failed: {e}")

    del feat; gc.collect()

# =============================================================================
# UPLOAD
# =============================================================================
print("\n[upload] uploading figures to HF Hub")
aggregate_paths = [fig03_path + ".pdf", fig05_path + ".pdf", fig0607_path + ".pdf"]
for path in aggregate_paths:
    name = os.path.basename(path)
    try:
        upload_file(path_or_fileobj=path, path_in_repo=f"viz_paper/{name}",
                    repo_id=HF_REPO, token=HF_TOKEN)
        print(f"  -> viz_paper/{name}")
    except Exception as e:
        print(f"  upload failed for {name}: {e}")

for ds_key, figmap in per_sample_results.items():
    for fname, paths in figmap.items():
        for path in paths:
            name = os.path.basename(path)
            try:
                upload_file(path_or_fileobj=path, path_in_repo=f"viz_paper/{ds_key}/{name}",
                            repo_id=HF_REPO, token=HF_TOKEN)
            except Exception as e:
                print(f"  upload failed {name}: {e}")

summary = {
    "global_mae": mae_global, "global_pearson": corr_global,
    "per_dataset_mae": {k: float(v) for k, v in zip(ds_keys, ds_maes)},
    "per_dataset_n":   {k: int(len(per_ds_rho[k][0])) for k in ds_keys},
    "kept_token_mean": {k: float(per_ds_mask[k].mean()) for k in ds_keys},
    "kept_token_std":  {k: float(per_ds_mask[k].std())  for k in ds_keys},
    "max_sens_mean":   {k: float(per_ds_sens[k].mean()) for k in ds_keys},
    "student_n_params": int(sum(p.numel() for p in student.parameters())),
}
sum_path = os.path.join(OUT_DIR, "summary_paper.json")
with open(sum_path, "w") as f:
    json.dump(summary, f, indent=2)
try:
    upload_file(path_or_fileobj=sum_path, path_in_repo="viz_paper/summary.json",
                repo_id=HF_REPO, token=HF_TOKEN)
except Exception as e:
    print(f"  upload failed for summary.json: {e}")

print(f"\n[done] MAE={mae_global:.4f}  Pearson r={corr_global:.4f}  "
      f"student params={summary['student_n_params']:,}")
