"""
module_01_data_prep.py
======================

Step 1 of the pipeline.

Loads each VQA dataset and precomputes per-sample:
  - CLIP-ViT-L/14@336 patch features  (576 tokens, 1024-dim)
  - CLIP-ViT-L/14@336 pooled CLS      (1024-dim)
  - CLIP-ViT-L/14@336 text embedding  (768-dim)
  - DINOv2-ViT-S/14 patch features    (256 tokens, 384-dim)

Saves one artifact per dataset to the HF Hub.

Usage
-----
Set DATASET_KEY to one of: vqav2 gqa okvqa slake vqarad pathvqa rsvqalr
Set HF_USER, HF_TOKEN, and HF_REPO as environment variables, then run:

    python module_01_data_prep.py
"""

import os
import gc
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from transformers import (
    CLIPVisionModel, CLIPImageProcessor,
    CLIPTextModel, CLIPTokenizer,
    AutoModel, AutoImageProcessor,
)
from huggingface_hub import login, create_repo, upload_file

# =============================================================================
# CONFIG  — override via environment variables or edit here
# =============================================================================
HF_USER     = os.environ.get("HF_USER",  "")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_REPO     = os.environ.get("HF_REPO",  f"{HF_USER}/QprivVL")

DATASET_KEY = os.environ.get("DATASET_KEY", "vqarad")
MAX_SAMPLES = 3000
BATCH_SIZE  = 64
IMAGE_SIZE  = 336   # CLIP-L/14@336 -> 576 patch tokens

CLIP_VISION = "openai/clip-vit-large-patch14-336"
CLIP_TEXT   = "openai/clip-vit-large-patch14-336"
DINO_MODEL  = "facebook/dinov2-small"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
OUT_NAME = f"features_{DATASET_KEY}.pt"

assert HF_TOKEN, "Set HF_TOKEN before running."
assert HF_USER,  "Set HF_USER before running."

# =============================================================================
# DATASET LOADERS
# =============================================================================

def _to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, dict) and "bytes" in x and x["bytes"]:
        import io
        return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise ValueError(f"Unrecognized image type: {type(x)}")


def _flatten_answer(a):
    if a is None:
        return ""
    if isinstance(a, list):
        if not a:
            return ""
        a = a[0]
        if isinstance(a, dict):
            return str(a.get("answer", ""))
        return str(a)
    return str(a)


def load_vqav2(max_samples):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/VQAv2", split="validation")
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return [{"image":    _to_pil(e["image"]),
             "question": str(e.get("question", "")),
             "answer":   _flatten_answer(e.get("multiple_choice_answer") or e.get("answers")),
             "qtype":    str(e.get("question_type", "unknown"))} for e in ds]


def load_gqa(max_samples):
    from datasets import load_dataset
    instr  = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    images = load_dataset("lmms-lab/GQA", "testdev_balanced_images",       split="testdev")
    img_lookup = {row["id"]: row["image"] for row in images}
    if len(instr) > max_samples:
        instr = instr.select(range(max_samples))
    out = []
    for ex in instr:
        img = img_lookup.get(ex["imageId"])
        if img is None:
            continue
        out.append({"image":    _to_pil(img),
                    "question": str(ex.get("question", "")),
                    "answer":   _flatten_answer(ex.get("answer")),
                    "qtype":    str((ex.get("types") or {}).get("structural", "unknown"))})
    return out


def load_okvqa(max_samples):
    from datasets import load_dataset
    try:
        ds = load_dataset("lmms-lab/OK-VQA", split="val2014")
    except Exception:
        ds = load_dataset("lmms-lab/OK-VQA", split="validation")
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return [{"image":    _to_pil(e["image"]),
             "question": str(e.get("question", "")),
             "answer":   _flatten_answer(e.get("answers")),
             "qtype":    str(e.get("question_type", "unknown"))} for e in ds]


def load_slake(max_samples):
    from datasets import load_dataset
    ds = load_dataset("mdwiratathya/SLAKE-vqa-english", split="train")
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return [{"image":    _to_pil(e["image"]),
             "question": str(e.get("question", "")),
             "answer":   _flatten_answer(e.get("answer")),
             "qtype":    "medical"} for e in ds]


def load_vqarad(max_samples):
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/vqa-rad", split="train")
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return [{"image":    _to_pil(e["image"]),
             "question": str(e.get("question", "")),
             "answer":   _flatten_answer(e.get("answer")),
             "qtype":    "radiology"} for e in ds]


def load_pathvqa(max_samples):
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/path-vqa", split="train")
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return [{"image":    _to_pil(e["image"]),
             "question": str(e.get("question", "")),
             "answer":   _flatten_answer(e.get("answer")),
             "qtype":    "pathology"} for e in ds]


def load_rsvqalr(max_samples):
    from datasets import load_dataset
    ds = load_dataset("exibings/rsvqa-lr", split="train")
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return [{"image":    _to_pil(e["image"]),
             "question": str(e.get("question", "")),
             "answer":   _flatten_answer(e.get("answer")),
             "qtype":    str(e.get("type", "unknown"))} for e in ds]


LOADERS = {
    "vqav2":   load_vqav2,
    "gqa":     load_gqa,
    "okvqa":   load_okvqa,
    "slake":   load_slake,
    "vqarad":  load_vqarad,
    "pathvqa": load_pathvqa,
    "rsvqalr": load_rsvqalr,
}

# =============================================================================
# SETUP
# =============================================================================
login(token=HF_TOKEN)
create_repo(HF_REPO, exist_ok=True, private=True, repo_type="model")
print(f"[setup] HF repo: {HF_REPO}")

# =============================================================================
# LOAD DATA
# =============================================================================
print(f"[data] loading {DATASET_KEY!r} (max {MAX_SAMPLES})")
samples = LOADERS[DATASET_KEY](MAX_SAMPLES)
print(f"[data] {len(samples)} samples")
assert samples, f"No samples loaded for {DATASET_KEY}"

# =============================================================================
# LOAD MODELS
# =============================================================================
print(f"[models] CLIP vision {CLIP_VISION}")
clip_vision   = CLIPVisionModel.from_pretrained(CLIP_VISION).to(DEVICE).eval()
clip_img_proc = CLIPImageProcessor.from_pretrained(CLIP_VISION)

print(f"[models] CLIP text {CLIP_TEXT}")
clip_text_enc = CLIPTextModel.from_pretrained(CLIP_TEXT).to(DEVICE).eval()
clip_tok      = CLIPTokenizer.from_pretrained(CLIP_TEXT)

print(f"[models] DINOv2 {DINO_MODEL}")
dino      = AutoModel.from_pretrained(DINO_MODEL).to(DEVICE).eval()
dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL)

for p in list(clip_vision.parameters()) + list(clip_text_enc.parameters()) + list(dino.parameters()):
    p.requires_grad = False

assert clip_text_enc.config.hidden_size == 768, (
    f"Expected clip_text hidden_size=768 but got {clip_text_enc.config.hidden_size}. "
    f"Check CLIP_TEXT."
)
print(f"[models] dims: CLIP-vis={clip_vision.config.hidden_size}  "
      f"CLIP-txt={clip_text_enc.config.hidden_size}  DINO={dino.config.hidden_size}")

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_batch(images, qs):
    # CLIP vision
    cx   = clip_img_proc(images=images, return_tensors="pt").pixel_values.to(DEVICE)
    cout = clip_vision(pixel_values=cx, output_hidden_states=True)
    c_hid      = cout.hidden_states[-2]
    clip_cls   = c_hid[:, 0].cpu()
    clip_patch = c_hid[:, 1:].cpu()  # (B, 576, 1024)
    # CLIP text
    tk       = clip_tok(qs, padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
    clip_txt = clip_text_enc(**tk).pooler_output.cpu()  # (B, 768)
    # DINOv2
    dx         = dino_proc(images=images, return_tensors="pt").pixel_values.to(DEVICE)
    dino_patch = dino(pixel_values=dx).last_hidden_state[:, 1:].cpu()  # (B, 256, 384)
    return clip_cls, clip_patch, clip_txt, dino_patch


clip_pooled_all, clip_patches_all, clip_text_all, dino_patches_all = [], [], [], []
questions, answers, qtypes, idxs = [], [], [], []
batch_imgs, batch_qs, batch_a, batch_qt, batch_i = [], [], [], [], []

for i, item in enumerate(tqdm(samples, desc=f"features[{DATASET_KEY}]")):
    batch_imgs.append(item["image"]); batch_qs.append(item["question"])
    batch_a.append(item["answer"]);   batch_qt.append(item["qtype"]); batch_i.append(i)
    if len(batch_imgs) == BATCH_SIZE:
        ccls, cpat, ctxt, dpat = extract_batch(batch_imgs, batch_qs)
        clip_pooled_all.append(ccls); clip_patches_all.append(cpat)
        clip_text_all.append(ctxt);   dino_patches_all.append(dpat)
        questions.extend(batch_qs); answers.extend(batch_a)
        qtypes.extend(batch_qt);    idxs.extend(batch_i)
        batch_imgs, batch_qs, batch_a, batch_qt, batch_i = [], [], [], [], []

if batch_imgs:
    ccls, cpat, ctxt, dpat = extract_batch(batch_imgs, batch_qs)
    clip_pooled_all.append(ccls); clip_patches_all.append(cpat)
    clip_text_all.append(ctxt);   dino_patches_all.append(dpat)
    questions.extend(batch_qs); answers.extend(batch_a)
    qtypes.extend(batch_qt);    idxs.extend(batch_i)

# =============================================================================
# SAVE AND UPLOAD
# =============================================================================
artifact = {
    "clip_pooled":  torch.cat(clip_pooled_all).half(),   # (N, 1024)
    "clip_patches": torch.cat(clip_patches_all).half(),  # (N, 576, 1024)
    "clip_text":    torch.cat(clip_text_all).half(),     # (N, 768)
    "dino_patches": torch.cat(dino_patches_all).half(),  # (N, 256, 384)
    "questions": questions, "answers": answers,
    "qtypes": qtypes, "idxs": idxs,
    "meta": {
        "dataset_key": DATASET_KEY, "n_samples": len(questions),
        "clip_vision": CLIP_VISION, "clip_text": CLIP_TEXT,
        "dino_model":  DINO_MODEL,  "image_size": IMAGE_SIZE,
    },
}
print(f"[shapes] clip_patches={tuple(artifact['clip_patches'].shape)}  "
      f"dino_patches={tuple(artifact['dino_patches'].shape)}  "
      f"clip_text={tuple(artifact['clip_text'].shape)}")

local_path = f"/tmp/{OUT_NAME}"
torch.save(artifact, local_path)
upload_file(path_or_fileobj=local_path, path_in_repo=OUT_NAME,
            repo_id=HF_REPO, token=HF_TOKEN)
print(f"[done] uploaded -> {HF_REPO}/{OUT_NAME}")

del clip_vision, clip_text_enc, dino
gc.collect(); torch.cuda.empty_cache()
