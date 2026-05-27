"""
module_04_student.py
====================

Step 4 of the pipeline.  Distils the teacher (module 3) into a lightweight
client-side Student that outputs:
  - rho_hat  (scalar pruning ratio per sample)
  - mask_prob (per-token keep probability over 576 CLIP patch positions)

Combined scoring: combined = alpha * utility - beta * sensitivity

Inputs  (from HF Hub): features_*.pt, rho_star_*.pt, sensitivity_head.pt,
                        teacher_predictor.pt
Outputs (to HF Hub):   student_model.pt

Usage
-----
    python module_04_student.py
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from transformers import AutoModel, AutoImageProcessor, CLIPTextModel, CLIPTokenizer
from huggingface_hub import login, hf_hub_download, upload_file

# =============================================================================
# CONFIG
# =============================================================================
HF_USER  = os.environ.get("HF_USER",  "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO  = os.environ.get("HF_REPO",  f"{HF_USER}/QprivVL")

DATASETS   = ["vqav2", "gqa", "okvqa", "slake", "vqarad", "pathvqa"]
DINO_MODEL = "facebook/dinov2-small"
CLIP_TEXT  = "openai/clip-vit-large-patch14-336"

ALPHA   = 1.0
BETA    = 1.0

STU_HIDDEN  = 256
STU_DEPTH   = 3
STU_DROPOUT = 0.1

EPOCHS     = 20
BATCH_SIZE = 128
LR         = 3e-4
WEIGHT_RHO  = 1.0
WEIGHT_MASK = 1.0
VAL_FRAC    = 0.1

DEVICE   = "cuda"
OUT_NAME = "student_model.pt"

assert HF_TOKEN and HF_USER, "Set HF_TOKEN and HF_USER before running."
login(token=HF_TOKEN)

# =============================================================================
# MODEL CLASSES (must match module 3 exactly for state-dict loading)
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
        self.Wq    = nn.Linear(txt_dim, proj_dim, bias=False)
        self.Wk    = nn.Linear(vis_dim, proj_dim, bias=False)
        self.scale = proj_dim ** 0.5

    def forward(self, visual, text):
        return torch.einsum("bd,bnd->bn", self.Wq(text), self.Wk(visual)) / self.scale


class Student(nn.Module):
    def __init__(self, vis_dim=1024, txt_dim=768, dino_dim=384,
                 teacher_cfg=None, sens_cfg=None, alpha=ALPHA, beta=BETA):
        super().__init__()
        self.sens_head = SensitivityHead(**(sens_cfg or {"in_dim": dino_dim, "hidden": 128}))
        self.predictor = ThresholdPredictor(**teacher_cfg)
        self.scorer    = QGTPScorer(vis_dim, txt_dim)
        self.gamma     = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("_alpha", torch.tensor(float(alpha)))
        self.register_buffer("_beta",  torch.tensor(float(beta)))

    @staticmethod
    def _align_spatial(sens_dino, clip_side=24, dino_side=16):
        B    = sens_dino.size(0)
        grid = sens_dino.view(B, 1, dino_side, dino_side)
        up   = F.interpolate(grid, size=(clip_side, clip_side),
                             mode="bilinear", align_corners=False)
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
        x_pred       = torch.cat([clip_pooled, clip_text, sens_sum], dim=-1)
        rho_hat      = self.predictor(x_pred)
        mask_prob    = torch.sigmoid(self.gamma * combined)
        return {"rho": rho_hat, "mask": mask_prob,
                "util": util_norm, "sens": sens_aligned, "combined": combined}


# =============================================================================
# LOAD TEACHER ARTIFACTS
# =============================================================================
print("[load] downloading teacher + sensitivity head")
teacher_art = torch.load(hf_hub_download(HF_REPO, "teacher_predictor.pt", token=HF_TOKEN),
                         map_location="cpu")
sens_art    = torch.load(hf_hub_download(HF_REPO, "sensitivity_head.pt",  token=HF_TOKEN),
                         map_location="cpu")
print(f"[load] teacher in_dim={teacher_art['config']['in_dim']}  "
      f"val_mse={teacher_art['meta']['val_mse']:.4f}")

# =============================================================================
# AGGREGATE TRAINING DATA
# =============================================================================
print("\n[data] aggregating across datasets")
all_clip_patches, all_clip_pooled, all_clip_text, all_dino_patches, all_rho_star = [], [], [], [], []

for key in DATASETS:
    try:
        feat = torch.load(hf_hub_download(HF_REPO, f"features_{key}.pt", token=HF_TOKEN),
                          map_location="cpu")
        rho  = torch.load(hf_hub_download(HF_REPO, f"rho_star_{key}.pt", token=HF_TOKEN),
                          map_location="cpu")
        idxs = rho["idxs"]
        all_clip_patches.append(feat["clip_patches"].float()[idxs])
        all_clip_pooled.append( feat["clip_pooled" ].float()[idxs])
        all_clip_text.append(   feat["clip_text"   ].float()[idxs])
        all_dino_patches.append(feat["dino_patches"].float()[idxs])
        all_rho_star.append(    rho["rho_star"])
        print(f"  {key}: {len(rho['rho_star'])} samples  clip_text_dim={feat['clip_text'].shape[-1]}")
    except Exception as e:
        print(f"  WARN skipping {key}: {e}")

CP  = torch.cat(all_clip_patches)
CPo = torch.cat(all_clip_pooled)
CT  = torch.cat(all_clip_text)
DP  = torch.cat(all_dino_patches)
RHO = torch.cat(all_rho_star)

assert CT.size(-1) == 768, (
    f"clip_text has dim {CT.size(-1)} but expects 768. Re-run module 1."
)
expected = CP.size(-1) + CT.size(-1) + 3
actual   = teacher_art["config"]["in_dim"]
assert expected == actual, (
    f"teacher_predictor in_dim={actual} but inputs need in_dim={expected}. "
    f"Re-run module 3."
)
print(f"[data] total {RHO.size(0)} samples  dims vis={CP.size(-1)} txt={CT.size(-1)} dino={DP.size(-1)}")

# =============================================================================
# TEACHER TARGETS (rho + per-token mask)
# =============================================================================
sens_teacher = SensitivityHead(**sens_art["config"]).to(DEVICE).eval()
sens_teacher.load_state_dict(sens_art["state_dict"])
for p in sens_teacher.parameters():
    p.requires_grad = False

teacher_pred = ThresholdPredictor(**teacher_art["config"]).to(DEVICE).eval()
teacher_pred.load_state_dict(teacher_art["state_dict"])
for p in teacher_pred.parameters():
    p.requires_grad = False

teacher_scorer = QGTPScorer(CP.size(-1), CT.size(-1)).to(DEVICE).eval()
for p in teacher_scorer.parameters():
    p.requires_grad = False


@torch.no_grad()
def teacher_targets(cp, cpo, ct, dp):
    sens_dino    = sens_teacher(dp)
    sens_aligned = Student._align_spatial(sens_dino)
    util_norm    = torch.sigmoid(teacher_scorer(cp, ct))
    combined     = ALPHA * util_norm - BETA * sens_aligned
    mask_target  = torch.sigmoid(3.0 * combined)
    sens_sum     = Student._sens_summary(sens_dino)
    rho_target   = teacher_pred(torch.cat([cpo, ct, sens_sum], dim=-1))
    return rho_target, mask_target, combined


# =============================================================================
# BUILD AND TRAIN STUDENT
# =============================================================================
student = Student(
    vis_dim     = CP.size(-1),
    txt_dim     = CT.size(-1),
    dino_dim    = DP.size(-1),
    teacher_cfg = teacher_art["config"],
    sens_cfg    = sens_art["config"],
).to(DEVICE)
student.sens_head.load_state_dict(sens_art["state_dict"])
student.predictor.load_state_dict(teacher_art["state_dict"])
print(f"[student] {sum(p.numel() for p in student.parameters()):,} parameters")

rng   = np.random.RandomState(42)
perm  = rng.permutation(RHO.size(0))
n_val = max(1, int(VAL_FRAC * RHO.size(0)))
val_idx, train_idx = perm[:n_val], perm[n_val:]

opt   = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)


def run_epoch(mode):
    student.train(mode == "train")
    ids = train_idx if mode == "train" else val_idx
    if mode == "train":
        rng.shuffle(ids)
    total_rho, total_mask, steps = 0.0, 0.0, 0
    for s in range(0, len(ids), BATCH_SIZE):
        b   = ids[s:s+BATCH_SIZE]
        cp  = CP[b].to(DEVICE); cpo = CPo[b].to(DEVICE)
        ct  = CT[b].to(DEVICE); dp  = DP[b].to(DEVICE)
        rho_star = RHO[b].to(DEVICE)
        with torch.no_grad():
            rho_t, mask_t, _ = teacher_targets(cp, cpo, ct, dp)
        out      = student(cp, cpo, ct, dp)
        loss_rho  = 0.5 * F.mse_loss(out["rho"], rho_t) + 0.5 * F.mse_loss(out["rho"], rho_star)
        loss_mask = F.binary_cross_entropy(out["mask"], mask_t)
        loss      = WEIGHT_RHO * loss_rho + WEIGHT_MASK * loss_mask
        if mode == "train":
            opt.zero_grad(); loss.backward(); opt.step()
        total_rho  += loss_rho.item(); total_mask += loss_mask.item(); steps += 1
    return total_rho / steps, total_mask / steps


best_val = float("inf")
for ep in range(EPOCHS):
    tr_rho, tr_mask = run_epoch("train")
    vl_rho, vl_mask = run_epoch("val")
    sched.step()
    vl_total = vl_rho + vl_mask
    flag = "*" if vl_total < best_val else " "
    best_val = min(best_val, vl_total)
    print(f"[student] ep {ep:2d}{flag}  "
          f"tr_rho={tr_rho:.4f} tr_mask={tr_mask:.4f}  "
          f"vl_rho={vl_rho:.4f} vl_mask={vl_mask:.4f}")

# =============================================================================
# SAVE AND UPLOAD
# =============================================================================
n_params = sum(p.numel() for p in student.parameters())
student_artifact = {
    "state_dict": {k: v.cpu() for k, v in student.state_dict().items()},
    "config": {
        "vis_dim": CP.size(-1), "txt_dim": CT.size(-1), "dino_dim": DP.size(-1),
        "teacher_cfg": teacher_art["config"], "sens_cfg": sens_art["config"],
        "alpha": ALPHA, "beta": BETA,
    },
    "meta": {
        "n_params": n_params, "n_train": len(train_idx), "n_val": len(val_idx),
        "best_val_total": best_val, "dino_model": DINO_MODEL, "clip_text": CLIP_TEXT,
        "datasets": DATASETS,
    },
}
local_path = f"/tmp/{OUT_NAME}"
torch.save(student_artifact, local_path)
upload_file(path_or_fileobj=local_path, path_in_repo=OUT_NAME,
            repo_id=HF_REPO, token=HF_TOKEN)
print(f"[student] uploaded -> {HF_REPO}/{OUT_NAME}  ({n_params/1e6:.2f}M params)")

gc.collect(); torch.cuda.empty_cache()
