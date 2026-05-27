"""
module_02_rho_search.py
=======================

Step 2 of the pipeline.

For each sample, finds the smallest pruning ratio rho_star such that LLaVA
still produces the correct answer when only the top-k question-relevant
visual tokens are kept.  This gives the per-sample utility floor used by
the teacher in module 3.

Inputs  (from HF Hub): features_<dataset>.pt  (module 1)
Outputs (to HF Hub):   rho_star_<dataset>.pt

Usage
-----
    python module_02_rho_search.py
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from transformers import LlavaForConditionalGeneration, AutoProcessor
from huggingface_hub import login, hf_hub_download, upload_file

# =============================================================================
# CONFIG
# =============================================================================
HF_USER  = os.environ.get("HF_USER",  "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO  = os.environ.get("HF_REPO",  f"{HF_USER}/QprivVL")

DATASETS = ["vqav2", "gqa", "okvqa", "slake", "vqarad", "pathvqa"]

LLAVA_ID        = "llava-hf/llava-1.5-7b-hf"
RHO_GRID        = [0.1, 0.2, 0.3, 0.5, 0.7]
MAX_PER_DATASET = 500
GEN_MAX_NEW     = 10
DEVICE          = "cuda"

assert HF_TOKEN and HF_USER, "Set HF_TOKEN and HF_USER before running."

# =============================================================================
# QGTP SCORER (random-init, used as a fixed ranking function only)
# =============================================================================

class QGTPScorer(nn.Module):
    """Cross-attention utility scorer (vis_dim, txt_dim inferred at runtime)."""

    def __init__(self, vis_dim, txt_dim, proj_dim=256):
        super().__init__()
        self.Wq    = nn.Linear(txt_dim, proj_dim, bias=False)
        self.Wk    = nn.Linear(vis_dim, proj_dim, bias=False)
        self.scale = proj_dim ** 0.5

    def forward(self, visual, text, rho):
        q      = self.Wq(text)
        k      = self.Wk(visual)
        scores = torch.einsum("bd,bnd->bn", q, k) / self.scale
        B, N   = scores.shape
        out    = []
        for b in range(B):
            kb = max(1, int((1 - rho[b].item()) * N))
            _, idx = torch.topk(scores[b], kb)
            out.append(torch.sort(idx).values)
        return out

# =============================================================================
# LOAD LLaVA
# =============================================================================
print(f"[models] loading {LLAVA_ID}")
llava     = LlavaForConditionalGeneration.from_pretrained(
    LLAVA_ID, torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(LLAVA_ID)
llava.eval()
for p in llava.parameters():
    p.requires_grad = False

# =============================================================================
# LLaVA INFERENCE WITH PRE-COMPUTED VISUAL TOKENS
# =============================================================================

def _get_inner(model, attr):
    if hasattr(model, attr):
        return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, attr):
        return getattr(model.model, attr)
    raise AttributeError(f"Cannot find {attr} on LLaVA model")


@torch.no_grad()
def llava_answer(patch_tokens, question, max_new=GEN_MAX_NEW):
    projector      = _get_inner(llava, "multi_modal_projector")
    language_model = _get_inner(llava, "language_model")
    vis_feat = projector(patch_tokens.unsqueeze(0).to(DEVICE).to(torch.bfloat16))
    prompt   = f"USER: <image>\n{question}\nASSISTANT:"
    enc      = processor.tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = enc.input_ids

    text_embeds = language_model.get_input_embeddings()(input_ids)
    img_tok     = llava.config.image_token_index
    mask        = (input_ids == img_tok)
    positions   = mask.nonzero(as_tuple=True)[1]
    new_embeds  = torch.cat([
        text_embeds[:, :positions[0]],
        vis_feat,
        text_embeds[:, positions[-1] + 1:],
    ], dim=1)
    attention_mask = torch.ones(new_embeds.shape[:2], dtype=torch.long,
                                device=new_embeds.device)
    out = llava.generate(
        inputs_embeds=new_embeds, attention_mask=attention_mask,
        max_new_tokens=max_new, do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()


def answer_matches(pred: str, gold: str) -> bool:
    g = gold.strip().lower()
    return bool(g) and g.split()[0] in pred

# =============================================================================
# PER-DATASET SEARCH
# =============================================================================

def search_dataset(dataset_key: str):
    feat_name = f"features_{dataset_key}.pt"
    out_name  = f"rho_star_{dataset_key}.pt"

    print(f"\n[data] downloading {feat_name}")
    local = hf_hub_download(repo_id=HF_REPO, filename=feat_name, token=HF_TOKEN)
    art   = torch.load(local, map_location="cpu")

    clip_patches = art["clip_patches"].float()
    clip_text    = art["clip_text"].float()
    questions    = art["questions"]
    answers      = art["answers"]

    txt_dim = clip_text.size(-1)
    if txt_dim != 768:
        print(f"  WARN: clip_text dim={txt_dim} (expected 768). Re-run module 1.")

    N = min(MAX_PER_DATASET, len(answers))
    print(f"[data] searching {N}/{len(answers)} samples  txt_dim={txt_dim}")

    scorer = QGTPScorer(vis_dim=clip_patches.size(-1), txt_dim=txt_dim).to(DEVICE)

    rho_star, kept_idxs = [], []
    for i in tqdm(range(N), desc=f"rho[{dataset_key}]"):
        visual = clip_patches[i:i+1].to(DEVICE)
        text   = clip_text[i:i+1].to(DEVICE)
        gold   = answers[i]
        chosen = 1.0
        for rho_val in sorted(RHO_GRID):
            rho_t   = torch.tensor([rho_val], device=DEVICE)
            pruned  = visual[0, scorer(visual, text, rho_t)[0]]
            pred    = llava_answer(pruned, questions[i])
            if answer_matches(pred, gold):
                chosen = rho_val
                break
        rho_star.append(chosen)
        kept_idxs.append(i)

    artifact = {
        "idxs":     kept_idxs,
        "rho_star": torch.tensor(rho_star, dtype=torch.float32),
        "rho_grid": RHO_GRID,
        "meta": {
            "features_from": feat_name, "llava": LLAVA_ID,
            "n": N, "dataset_key": dataset_key, "txt_dim": txt_dim,
        },
    }
    local_out = f"/tmp/{out_name}"
    torch.save(artifact, local_out)
    upload_file(path_or_fileobj=local_out, path_in_repo=out_name,
                repo_id=HF_REPO, token=HF_TOKEN)
    print(f"[done] {dataset_key}: mean rho*={sum(rho_star)/len(rho_star):.3f}  "
          f"uploaded -> {HF_REPO}/{out_name}")

# =============================================================================
# MAIN
# =============================================================================
login(token=HF_TOKEN)
for k in DATASETS:
    try:
        search_dataset(k)
    except Exception as e:
        print(f"[warn] {k} failed: {e}. Continuing.")

del llava
gc.collect(); torch.cuda.empty_cache()
print("\n[done] rho* search complete.")
