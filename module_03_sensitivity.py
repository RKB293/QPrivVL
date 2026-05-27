"""
module_03_sensitivity.py
========================

Step 3 of the pipeline.  Trains two teacher components:

  (A) SensitivityHead  — DINOv2 patch features → per-patch privacy scores
      Trained with image-level weak labels via multiple-instance learning.

  (B) ThresholdPredictor — (clip_pooled, clip_text, sensitivity_summary)
      → scalar rho_hat, trained to match rho_star from module 2.

Inputs  (from HF Hub): features_<dataset>.pt, rho_star_<dataset>.pt
Outputs (to HF Hub):   sensitivity_head.pt, teacher_predictor.pt

Usage
-----
    python module_03_sensitivity.py
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor
from huggingface_hub import login, hf_hub_download, upload_file

# =============================================================================
# CONFIG
# =============================================================================
HF_USER  = os.environ.get("HF_USER",  "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO  = os.environ.get("HF_REPO",  f"{HF_USER}/QprivVL")

DATASETS_FOR_TEACHER = ["vqav2", "gqa", "okvqa", "slake", "vqarad", "pathvqa"]
DINO_MODEL = "facebook/dinov2-small"

PRIVACY_SOURCES = {
    "celeba_pos":  ("tpremoli/CelebA-attrs",                    "train", 400, 1),
    "celeba_neg":  ("mnist",                                    "train", 400, 0),
    "mimic_pos":   ("Sohaibsoussi/NIH-Chest-X-ray-dataset-small","train", 400, 1),
    "rsvqa_pos":   ("blanchon/EuroSAT_RGB",                     "train", 400, 1),
    "generic_neg": ("cifar10",                                  "train", 400, 0),
}

BATCH_SIZE      = 64
EPOCHS_SENS     = 10
LR_SENS         = 3e-4
VAL_FRAC        = 0.1
EPOCHS_TEACHER  = 20
LR_TEACHER      = 1e-4
TEACHER_HIDDEN  = 512
TEACHER_LAYERS  = 3
TEACHER_DROPOUT = 0.1
DEVICE          = "cuda"

SENS_OUT_NAME    = "sensitivity_head.pt"
TEACHER_OUT_NAME = "teacher_predictor.pt"

assert HF_TOKEN and HF_USER, "Set HF_TOKEN and HF_USER before running."
login(token=HF_TOKEN)

# =============================================================================
# (A) SENSITIVITY HEAD — per-patch MIL classifier over DINOv2 features
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


print(f"[models] loading DINOv2 {DINO_MODEL}")
dino      = AutoModel.from_pretrained(DINO_MODEL).to(DEVICE).eval()
dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
for p in dino.parameters():
    p.requires_grad = False
D_DINO = dino.config.hidden_size
print(f"[models] DINOv2 dim={D_DINO}")


@torch.no_grad()
def dino_encode(images, batch=BATCH_SIZE):
    feats = []
    for s in range(0, len(images), batch):
        px  = dino_proc(images=images[s:s+batch], return_tensors="pt").pixel_values.to(DEVICE)
        out = dino(pixel_values=px).last_hidden_state[:, 1:]
        feats.append(out.cpu())
    return torch.cat(feats)


def safe_image(ex):
    for key in ("image", "img", "Image"):
        v = ex.get(key) if isinstance(ex, dict) else getattr(ex, key, None)
        if v is not None:
            try:
                return (v if isinstance(v, Image.Image) else Image.open(v)).convert("RGB")
            except Exception:
                pass
    return None


print("[sens] collecting privacy training samples")
sens_images, sens_labels = [], []
for src_key, (ds_id, split, n_max, label) in PRIVACY_SOURCES.items():
    try:
        print(f"  loading {src_key}: {ds_id}")
        try:
            ds = load_dataset(ds_id, split=split)
        except Exception as e1:
            if "Dataset scripts are no longer supported" in str(e1):
                ds = load_dataset(ds_id, split=split, revision="refs/convert/parquet")
            else:
                raise
        if len(ds) > n_max:
            ds = ds.select(range(n_max))
        for ex in ds:
            img = safe_image(ex)
            if img:
                sens_images.append(img)
                sens_labels.append(label)
    except Exception as e:
        print(f"  WARN: skipping {src_key}: {e}")

print(f"[sens] {len(sens_images)} samples  "
      f"pos={sum(sens_labels)}  neg={len(sens_labels)-sum(sens_labels)}")

print("[sens] encoding with DINOv2")
sens_feats = dino_encode(sens_images).float()
sens_y     = torch.tensor(sens_labels, dtype=torch.float32)

rng  = np.random.RandomState(42)
perm = rng.permutation(len(sens_y))
n_val       = max(1, int(VAL_FRAC * len(sens_y)))
val_idx, train_idx = perm[:n_val], perm[n_val:]

print("[sens] training head")
sens_head = SensitivityHead(in_dim=D_DINO).to(DEVICE)
opt_sens  = torch.optim.AdamW(sens_head.parameters(), lr=LR_SENS, weight_decay=1e-4)


def sens_epoch(mode="train"):
    sens_head.train(mode == "train")
    ids = train_idx if mode == "train" else val_idx
    if mode == "train":
        rng.shuffle(ids)
    total_loss, total_acc, n = 0.0, 0.0, 0
    for s in range(0, len(ids), BATCH_SIZE):
        b          = ids[s:s+BATCH_SIZE]
        X          = sens_feats[b].to(DEVICE)
        y          = sens_y[b].to(DEVICE)
        img_prob   = sens_head(X).max(dim=-1).values
        loss       = F.binary_cross_entropy(img_prob, y)
        if mode == "train":
            opt_sens.zero_grad(); loss.backward(); opt_sens.step()
        total_loss += loss.item() * len(b)
        total_acc  += ((img_prob > 0.5).float() == y).sum().item()
        n          += len(b)
    return total_loss / n, total_acc / n


for ep in range(EPOCHS_SENS):
    tl, ta = sens_epoch("train")
    vl, va = sens_epoch("val")
    print(f"[sens] ep {ep:2d}  train_loss={tl:.4f} acc={ta:.3f}  "
          f"val_loss={vl:.4f} acc={va:.3f}")

sens_artifact = {
    "state_dict": {k: v.cpu() for k, v in sens_head.state_dict().items()},
    "config":     {"in_dim": D_DINO, "hidden": 128},
    "meta":       {"n_train": len(train_idx), "val_acc": va, "dino_model": DINO_MODEL},
}
torch.save(sens_artifact, f"/tmp/{SENS_OUT_NAME}")
upload_file(path_or_fileobj=f"/tmp/{SENS_OUT_NAME}", path_in_repo=SENS_OUT_NAME,
            repo_id=HF_REPO, token=HF_TOKEN)
print(f"[sens] uploaded -> {HF_REPO}/{SENS_OUT_NAME}")

del sens_images, sens_feats
gc.collect(); torch.cuda.empty_cache()

# =============================================================================
# (B) THRESHOLD PREDICTOR — predicts scalar rho from utility + privacy signals
# =============================================================================

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


def sensitivity_summary(sens_patch_probs):
    return torch.cat([
        sens_patch_probs.mean(-1, keepdim=True),
        sens_patch_probs.max(-1, keepdim=True).values,
        (sens_patch_probs > 0.5).float().mean(-1, keepdim=True),
    ], dim=-1)


sens_head_eval = SensitivityHead(in_dim=D_DINO).to(DEVICE).eval()
sens_head_eval.load_state_dict(sens_head.state_dict())
for p in sens_head_eval.parameters():
    p.requires_grad = False

print("\n[teacher] aggregating training data across datasets")
X_all, y_all = [], []
observed_text_dims = set()

for key in DATASETS_FOR_TEACHER:
    try:
        feat_path = hf_hub_download(repo_id=HF_REPO, filename=f"features_{key}.pt", token=HF_TOKEN)
        rho_path  = hf_hub_download(repo_id=HF_REPO, filename=f"rho_star_{key}.pt",  token=HF_TOKEN)
        feat = torch.load(feat_path, map_location="cpu")
        rho  = torch.load(rho_path,  map_location="cpu")
        idxs        = rho["idxs"]
        rho_star    = rho["rho_star"]
        clip_pooled = feat["clip_pooled"].float()[idxs]
        clip_text   = feat["clip_text"].float()[idxs]
        dino_patch  = feat["dino_patches"].float()[idxs]
        observed_text_dims.add(int(clip_text.size(-1)))

        sens_summaries = []
        for s in range(0, len(rho_star), BATCH_SIZE):
            chunk = dino_patch[s:s+BATCH_SIZE].to(DEVICE)
            with torch.no_grad():
                sp = sens_head_eval(chunk)
            sens_summaries.append(sensitivity_summary(sp).cpu())
        sens_sum = torch.cat(sens_summaries)

        X = torch.cat([clip_pooled, clip_text, sens_sum], dim=-1)
        X_all.append(X); y_all.append(rho_star)
        print(f"  {key}: {len(X)} samples  clip_text_dim={clip_text.size(-1)}")
    except Exception as e:
        print(f"  WARN skipping {key}: {e}")

if len(observed_text_dims) > 1:
    raise RuntimeError(
        f"Inconsistent clip_text dims across datasets: {sorted(observed_text_dims)}. "
        f"Re-run module 1 for all datasets with the same CLIP_TEXT."
    )

X_all = torch.cat(X_all); y_all = torch.cat(y_all)
print(f"[teacher] total: X={tuple(X_all.shape)}  y={tuple(y_all.shape)}")

perm  = rng.permutation(len(y_all))
n_val = max(1, int(VAL_FRAC * len(y_all)))
val_idx, train_idx = perm[:n_val], perm[n_val:]

teacher = ThresholdPredictor(
    in_dim=X_all.size(-1), hidden=TEACHER_HIDDEN,
    n_layers=TEACHER_LAYERS, dropout=TEACHER_DROPOUT,
).to(DEVICE)
opt_t = torch.optim.AdamW(teacher.parameters(), lr=LR_TEACHER, weight_decay=0.01)

Xtr, ytr = X_all[train_idx].to(DEVICE), y_all[train_idx].to(DEVICE)
Xvl, yvl = X_all[val_idx].to(DEVICE),   y_all[val_idx].to(DEVICE)

for ep in range(EPOCHS_TEACHER):
    teacher.train()
    order = torch.randperm(len(ytr), device=DEVICE)
    tr_loss, steps = 0.0, 0
    for s in range(0, len(order), BATCH_SIZE):
        b    = order[s:s+BATCH_SIZE]
        loss = F.mse_loss(teacher(Xtr[b]), ytr[b])
        opt_t.zero_grad(); loss.backward(); opt_t.step()
        tr_loss += loss.item(); steps += 1
    teacher.eval()
    with torch.no_grad():
        val_mse = F.mse_loss(teacher(Xvl), yvl).item()
    print(f"[teacher] ep {ep:2d}  train_mse={tr_loss/steps:.4f}  val_mse={val_mse:.4f}")

teacher_artifact = {
    "state_dict": {k: v.cpu() for k, v in teacher.state_dict().items()},
    "config":     {"in_dim": X_all.size(-1), "hidden": TEACHER_HIDDEN,
                   "n_layers": TEACHER_LAYERS, "dropout": TEACHER_DROPOUT},
    "meta": {
        "datasets": DATASETS_FOR_TEACHER, "n_train": len(train_idx), "n_val": len(val_idx),
        "val_mse": val_mse, "sens_head": SENS_OUT_NAME,
        "clip_text_dim": sorted(observed_text_dims)[0],
    },
}
torch.save(teacher_artifact, f"/tmp/{TEACHER_OUT_NAME}")
upload_file(path_or_fileobj=f"/tmp/{TEACHER_OUT_NAME}", path_in_repo=TEACHER_OUT_NAME,
            repo_id=HF_REPO, token=HF_TOKEN)
print(f"[teacher] uploaded -> {HF_REPO}/{TEACHER_OUT_NAME}")

gc.collect(); torch.cuda.empty_cache()
print("\n[module-3 done]")
