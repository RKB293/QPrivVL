"""
qgtp_lib.py
===========

Shared infrastructure for the QGTP pipeline (modules 6-14):
  - QGTP operator (question-guided token pruning)
  - Student model loader (predicts pruning ratio rho)
  - LLaVA-1.5-7B wrapper with LoRA fine-tuning and generation
  - Dataset loaders and train/val/test split helpers
  - Per-round evaluation
  - HF Hub I/O helpers
"""

import os
import io
import gc
import re
import json
import math
import copy
import random
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Tuple

_HF_HOME = os.environ.get("HF_HOME", "")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if _HF_HOME:
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE",     os.path.join(_HF_HOME, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE",    os.path.join(_HF_HOME, "hub"))

if os.environ.get("OFFLINE") == "1":
    os.environ["HF_HUB_OFFLINE"]      = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import login, hf_hub_download, upload_file

from transformers import (
    AutoModel, AutoImageProcessor,
    CLIPVisionModel, CLIPImageProcessor,
    CLIPTextModel, CLIPTokenizer,
    LlavaForConditionalGeneration, AutoProcessor,
)

OFFLINE = (
    os.environ.get("HF_HUB_OFFLINE") == "1"
    or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
)

# =============================================================================
# CONFIG
# =============================================================================
HF_USER  = os.environ.get("HF_USER",  "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO  = os.environ.get("HF_REPO",  f"{HF_USER}/QprivVL") if HF_USER else ""

LLAVA_ID = "llava-hf/llava-1.5-7b-hf"
DINO_ID  = "facebook/dinov2-small"
CLIP_VID = "openai/clip-vit-large-patch14-336"
CLIP_TID = "openai/clip-vit-large-patch14-336"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =============================================================================
# STUDENT MODEL CLASSES
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
        q = self.Wq(text)
        k = self.Wk(visual)
        return torch.einsum("bd,bnd->bn", q, k) / self.scale


class Student(nn.Module):
    """Client-side student: outputs pruning ratio (rho) and keep-mask."""

    def __init__(self, vis_dim, txt_dim, dino_dim, teacher_cfg, sens_cfg,
                 alpha=1.0, beta=1.0):
        super().__init__()
        self.sens_head = SensitivityHead(**sens_cfg)
        self.predictor = ThresholdPredictor(**teacher_cfg)
        self.scorer    = QGTPScorer(vis_dim, txt_dim)
        self.gamma     = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("_alpha", torch.tensor(alpha))
        self.register_buffer("_beta",  torch.tensor(beta))

    @staticmethod
    def _align_spatial(sens_dino, clip_side=24, dino_side=16):
        B = sens_dino.size(0)
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
        util_raw     = self.scorer(clip_patches, clip_text)
        util_norm    = torch.sigmoid(util_raw)
        combined     = self._alpha * util_norm - self._beta * sens_aligned
        sens_sum     = self._sens_summary(sens_dino)
        x_pred       = torch.cat([clip_pooled, clip_text, sens_sum], dim=-1)
        rho_hat      = self.predictor(x_pred)
        mask_prob    = torch.sigmoid(self.gamma * combined)
        return {
            "rho": rho_hat, "mask": mask_prob,
            "util": util_norm, "sens": sens_aligned,
            "combined": combined,
        }


def load_student_from_hf():
    """Download student_model.pt from HF Hub and return a frozen module."""
    if not OFFLINE:
        if not HF_TOKEN:
            print("[student] WARN: HF_TOKEN not set.")
        else:
            login(token=HF_TOKEN)
    path = hf_hub_download(
        repo_id=HF_REPO, filename="student_model.pt",
        token=HF_TOKEN if HF_TOKEN else None,
        local_files_only=OFFLINE,
    )
    art = torch.load(path, map_location="cpu")
    cfg = art["config"]
    student = Student(
        vis_dim     = cfg["vis_dim"],
        txt_dim     = cfg["txt_dim"],
        dino_dim    = cfg["dino_dim"],
        teacher_cfg = cfg["teacher_cfg"],
        sens_cfg    = cfg["sens_cfg"],
        alpha       = cfg.get("alpha", 1.0),
        beta        = cfg.get("beta",  1.0),
    )
    student.load_state_dict(art["state_dict"])
    student.eval().to(DEVICE)
    for p in student.parameters():
        p.requires_grad = False
    print(f"[student] loaded {sum(p.numel() for p in student.parameters()):,} params")
    return student


# =============================================================================
# QGTP CONTROLLER
# =============================================================================

class QGTPController:
    """Wraps the visual token pruning policy.

    mode:
      'off'     – keep all 576 tokens (no pruning).
      'fixed'   – drop a fixed fraction `fixed_rho` of tokens, ranked by
                  utility score.
      'student' – use the trained student to predict rho per sample.
    """

    def __init__(self, mode: str = "student", fixed_rho: float = 0.5,
                 student: Optional[Student] = None,
                 use_mask_directly: bool = False,
                 min_keep: int = 16):
        assert mode in {"off", "student", "fixed"}
        self.mode      = mode
        self.fixed_rho = float(fixed_rho)
        self.student   = student
        self.use_mask  = use_mask_directly
        self.min_keep  = min_keep

    @torch.no_grad()
    def select(self, clip_patches, clip_pooled, clip_text, dino_patches):
        """Return a list of LongTensor index vectors (one per sample)."""
        B, N, _ = clip_patches.shape

        if self.mode == "off":
            return [torch.arange(N, device=clip_patches.device) for _ in range(B)]

        if self.mode == "fixed":
            k = max(self.min_keep, int(round((1.0 - self.fixed_rho) * N)))
            if self.student is not None:
                util = self.student.scorer(clip_patches, clip_text)
                return [torch.sort(torch.topk(util[b], k).indices).values
                        for b in range(B)]
            return [torch.linspace(0, N - 1, k, device=clip_patches.device).long()
                    for _ in range(B)]

        # mode == "student"
        assert self.student is not None, "student must be provided for mode='student'"
        out = self.student(clip_patches, clip_pooled, clip_text, dino_patches)
        idx_list = []
        if self.use_mask:
            for b in range(B):
                idx = (out["mask"][b] > 0.5).nonzero(as_tuple=True)[0]
                if idx.numel() < self.min_keep:
                    idx = out["mask"][b].topk(self.min_keep).indices
                idx_list.append(torch.sort(idx).values)
        else:
            util = out["util"]
            for b in range(B):
                rho_b = float(max(0.0, min(0.95, out["rho"][b].item())))
                k_b   = max(self.min_keep, int(round((1.0 - rho_b) * N)))
                idx   = torch.topk(util[b], k_b).indices
                idx_list.append(torch.sort(idx).values)
        return idx_list


# =============================================================================
# FROZEN ENCODERS (CLIP-V, CLIP-T, DINOv2)
# =============================================================================

class FrozenEncoders:
    """Holds the three frozen encoders on GPU."""

    def __init__(self):
        print("[encoders] loading frozen CLIP-V / CLIP-T / DINOv2 ...")
        self.clip_v   = CLIPVisionModel.from_pretrained(CLIP_VID).to(DEVICE).eval()
        self.clip_v_p = CLIPImageProcessor.from_pretrained(CLIP_VID)
        self.clip_t   = CLIPTextModel.from_pretrained(CLIP_TID).to(DEVICE).eval()
        self.clip_t_p = CLIPTokenizer.from_pretrained(CLIP_TID)
        self.dino     = AutoModel.from_pretrained(DINO_ID).to(DEVICE).eval()
        self.dino_p   = AutoImageProcessor.from_pretrained(DINO_ID)
        for m in (self.clip_v, self.clip_t, self.dino):
            for p in m.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode(self, images: List[Image.Image], questions: List[str]):
        cx   = self.clip_v_p(images=images, return_tensors="pt").pixel_values.to(DEVICE)
        cout = self.clip_v(pixel_values=cx, output_hidden_states=True)
        hid  = cout.hidden_states[-2]
        clip_pooled  = hid[:, 0]
        clip_patches = hid[:, 1:]     # (B, 576, 1024)

        tk        = self.clip_t_p(questions, padding=True, truncation=True,
                                  max_length=77, return_tensors="pt").to(DEVICE)
        clip_text = self.clip_t(**tk).pooler_output  # (B, 768)

        dx           = self.dino_p(images=images, return_tensors="pt").pixel_values.to(DEVICE)
        dino_patches = self.dino(pixel_values=dx).last_hidden_state[:, 1:]  # (B, 256, 384)

        return (clip_patches.float(), clip_pooled.float(),
                clip_text.float(), dino_patches.float())


# =============================================================================
# LLaVA WRAPPER WITH LoRA
# =============================================================================

def _get_inner(model, attr):
    """Locate an attribute at the top level or under model.model."""
    if hasattr(model, attr):
        return getattr(model, attr)
    if hasattr(model, "model") and hasattr(model.model, attr):
        return getattr(model.model, attr)
    raise AttributeError(f"Cannot find {attr} on LLaVA model")


DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


class LLaVAWithQGTP(nn.Module):
    """LLaVA-1.5-7B with LoRA adapters and QGTP token injection."""

    def __init__(self, lora_r: int = 16, lora_alpha: int = 32,
                 lora_targets: Tuple[str, ...] = DEFAULT_LORA_TARGETS,
                 lora_dropout: float = 0.05):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model

        print(f"[llava] loading {LLAVA_ID} (bf16)")
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_ID, torch_dtype=torch.bfloat16, device_map="balanced",
        )
        self.processor = AutoProcessor.from_pretrained(LLAVA_ID)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.image_token_id = int(self.llava.config.image_token_index)
        probe_ids = self.processor.tokenizer.encode("<image>", add_special_tokens=False)
        self._image_token_tokenizes = (
            len(probe_ids) == 1 and probe_ids[0] == self.image_token_id
        )

        for p in self.llava.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=list(lora_targets),
            lora_dropout=lora_dropout, bias="none",
        )
        lm = _get_inner(self.llava, "language_model")
        inject_adapter_in_model(lora_cfg, lm)

        for n, p in self.llava.named_parameters():
            p.requires_grad = ("lora_" in n)

        n_train = sum(p.numel() for p in self.llava.parameters() if p.requires_grad)
        print(f"[llava] LoRA trainable parameters: {n_train:,}")

    # ---- prompt helpers -------------------------------------------------------

    def _tokenize_prompt_with_image_slot(self, question: str):
        tok      = self.processor.tokenizer
        pre_ids  = tok("USER: ",             add_special_tokens=True,  return_tensors="pt").input_ids[0]
        post_ids = tok(f"\n{question}\nASSISTANT:", add_special_tokens=False, return_tensors="pt").input_ids[0]
        return pre_ids, post_ids

    def _tokenize_answer(self, answer: str):
        tok  = self.processor.tokenizer
        text = " " + answer.strip() + tok.eos_token
        return tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]

    # ---- shared embed construction -------------------------------------------

    def _build_embeds(self, patch_tokens_per_sample: List[torch.Tensor],
                      questions: List[str], device=DEVICE,
                      include_answer_for_loss: Optional[List[str]] = None):
        projector      = _get_inner(self.llava, "multi_modal_projector")
        language_model = _get_inner(self.llava, "language_model")
        embed_layer    = language_model.get_input_embeddings()

        proj_per_sample = []
        for v in patch_tokens_per_sample:
            proj = projector(v.unsqueeze(0).to(device).to(torch.bfloat16)).squeeze(0)
            proj_per_sample.append(proj)

        per_sample_embeds: List[torch.Tensor] = []
        per_sample_labels: List[torch.Tensor] = []

        for i, q in enumerate(questions):
            pre_ids, post_ids = self._tokenize_prompt_with_image_slot(q)
            pre_ids  = pre_ids.to(device)
            post_ids = post_ids.to(device)
            pre_emb  = embed_layer(pre_ids)
            post_emb = embed_layer(post_ids)
            img_emb  = proj_per_sample[i]

            if include_answer_for_loss is not None:
                ans_ids = self._tokenize_answer(include_answer_for_loss[i]).to(device)
                ans_emb = embed_layer(ans_ids)
                full_emb = torch.cat([pre_emb, img_emb, post_emb, ans_emb], dim=0)
                L_prompt = pre_emb.size(0) + img_emb.size(0) + post_emb.size(0)
                L_total  = full_emb.size(0)
                lbl = torch.full((L_total,), -100, dtype=torch.long, device=device)
                lbl[L_prompt:L_prompt + ans_ids.size(0)] = ans_ids
                per_sample_labels.append(lbl)
            else:
                full_emb = torch.cat([pre_emb, img_emb, post_emb], dim=0)
            per_sample_embeds.append(full_emb)

        max_len = max(e.size(0) for e in per_sample_embeds)
        D_lm    = per_sample_embeds[0].size(-1)
        B       = len(per_sample_embeds)
        is_gen  = include_answer_for_loss is None

        padded_embeds = torch.zeros(B, max_len, D_lm, device=device, dtype=torch.bfloat16)
        attn_mask     = torch.zeros(B, max_len, device=device, dtype=torch.long)
        labels        = None
        if not is_gen:
            labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)

        for i, e in enumerate(per_sample_embeds):
            L = e.size(0)
            if is_gen:
                padded_embeds[i, max_len - L:] = e
                attn_mask[i, max_len - L:]     = 1
            else:
                padded_embeds[i, :L] = e
                attn_mask[i, :L]     = 1
                labels[i, :L]        = per_sample_labels[i]

        return padded_embeds, attn_mask, labels

    # ---- forward / generate --------------------------------------------------

    def forward_loss(self, patch_tokens_per_sample, questions, answers):
        embeds, attn, labels = self._build_embeds(
            patch_tokens_per_sample, questions,
            include_answer_for_loss=answers,
        )
        return self.llava(inputs_embeds=embeds, attention_mask=attn, labels=labels).loss

    @torch.no_grad()
    def generate(self, patch_tokens_per_sample, questions, max_new_tokens=20):
        embeds, attn, _ = self._build_embeds(patch_tokens_per_sample, questions)
        out = self.llava.generate(
            inputs_embeds=embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return [self.processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
                for seq in out]

    def trainable_parameters(self):
        return [p for p in self.llava.parameters() if p.requires_grad]

    def lora_state_dict(self):
        sd = self.llava.state_dict()
        return OrderedDict((k, v.detach().cpu()) for k, v in sd.items() if "lora_" in k)

    def load_lora_state_dict(self, sd):
        cur = self.llava.state_dict()
        for k, v in sd.items():
            if k in cur:
                cur[k].copy_(v.to(cur[k].device).to(cur[k].dtype))


# =============================================================================
# DATASET LOADERS
# =============================================================================

def _to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, dict) and x.get("bytes"):
        return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")
    raise ValueError(f"unrecognized image type: {type(x)}")


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


def load_dataset_split(dataset_key: str, max_samples: int = 3000):
    """Load a VQA dataset as a flat list of dicts with keys:
       image (PIL), question (str), answer (str), qtype (str).
    """
    from datasets import load_dataset

    def _wrap(ds, img_key="image", q_key="question", a_key="answer", qtype="unknown"):
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        return [{"image": _to_pil(e[img_key]),
                 "question": str(e.get(q_key, "")),
                 "answer": _flatten_answer(e.get(a_key)),
                 "qtype": str(e.get("question_type", qtype))} for e in ds]

    if dataset_key == "vqav2":
        ds = load_dataset("lmms-lab/VQAv2", split="validation")
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        return [{"image": _to_pil(e["image"]),
                 "question": str(e.get("question", "")),
                 "answer": _flatten_answer(e.get("multiple_choice_answer") or e.get("answers")),
                 "qtype": str(e.get("question_type", "unknown"))} for e in ds]

    if dataset_key == "gqa":
        instr  = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
        images = load_dataset("lmms-lab/GQA", "testdev_balanced_images",       split="testdev")
        lookup = {row["id"]: row["image"] for row in images}
        if len(instr) > max_samples:
            instr = instr.select(range(max_samples))
        out = []
        for e in instr:
            img = lookup.get(e["imageId"])
            if img is None:
                continue
            out.append({"image": _to_pil(img), "question": str(e.get("question", "")),
                        "answer": _flatten_answer(e.get("answer")), "qtype": "gqa"})
        return out

    if dataset_key == "okvqa":
        try:
            ds = load_dataset("lmms-lab/OK-VQA", split="val2014")
        except Exception:
            ds = load_dataset("lmms-lab/OK-VQA", split="validation")
        return _wrap(ds, a_key="answers")

    if dataset_key == "slake":
        return _wrap(load_dataset("mdwiratathya/SLAKE-vqa-english", split="train"),
                     a_key="answer", qtype="medical")

    if dataset_key == "vqarad":
        return _wrap(load_dataset("flaviagiammarino/vqa-rad", split="train"),
                     a_key="answer", qtype="radiology")

    if dataset_key == "pathvqa":
        return _wrap(load_dataset("flaviagiammarino/path-vqa", split="train"),
                     a_key="answer", qtype="pathology")

    if dataset_key == "rsvqalr":
        return _wrap(load_dataset("exibings/rsvqa-lr", split="train"),
                     a_key="answer", qtype="remote-sensing")

    raise ValueError(f"unknown dataset_key: {dataset_key!r}")


def split_train_val_test(samples: List[dict],
                          train_frac=0.7, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)
    n    = len(samples)
    n_tr = int(train_frac * n)
    n_vl = int(val_frac * n)
    return samples[:n_tr], samples[n_tr:n_tr + n_vl], samples[n_tr + n_vl:]


def shard_for_clients(train_samples: List[dict], n_clients: int, seed=42):
    rng = random.Random(seed)
    samples = list(train_samples)
    rng.shuffle(samples)
    return [samples[i::n_clients] for i in range(n_clients)]


# =============================================================================
# EVALUATION
# =============================================================================

_PUNC_RE  = re.compile(r"[^\w\s]")
_WS_RE    = re.compile(r"\s+")
_ARTICLES = {"a", "an", "the"}


def _normalize_answer(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PUNC_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return " ".join(t for t in s.split() if t not in _ARTICLES)


def _f1_score(pred: str, gold: str) -> float:
    p = _normalize_answer(pred).split()
    g = _normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    n_common = sum(min(p.count(t), g.count(t)) for t in set(p) & set(g))
    if n_common == 0:
        return 0.0
    prec = n_common / len(p)
    rec  = n_common / len(g)
    return 2 * prec * rec / (prec + rec)


def vqa_match(pred: str, gold: str) -> float:
    p = _normalize_answer(pred)
    g = _normalize_answer(gold)
    if not g:
        return 0.0
    if g in {"yes", "no"}:
        first = p.split()[0] if p else ""
        return float(first == g)
    if p == g:
        return 1.0
    if (" " + g + " ") in (" " + p + " "):
        return 1.0
    return float(_f1_score(p, g) >= 0.5)


def loose_match(pred: str, gold: str) -> bool:
    return vqa_match(pred, gold) >= 1.0


@torch.no_grad()
def evaluate(llava: LLaVAWithQGTP, encoders: FrozenEncoders, qgtp: QGTPController,
             samples: List[dict], batch_size=4, max_eval: Optional[int] = None,
             desc="eval", max_new_tokens: int = 20) -> Dict[str, float]:
    if max_eval is not None and len(samples) > max_eval:
        samples = samples[:max_eval]
    n_correct = 0.0
    n_total   = 0
    avg_kept  = 0.0
    n_batches = 0
    pbar = tqdm(range(0, len(samples), batch_size), desc=desc, leave=False)
    for s in pbar:
        batch = samples[s:s + batch_size]
        imgs  = [b["image"]    for b in batch]
        qs    = [b["question"] for b in batch]
        golds = [b["answer"]   for b in batch]
        cp, cpo, ct, dp = encoders.encode(imgs, qs)
        keep_idxs = qgtp.select(cp, cpo, ct, dp)
        kept      = [cp[i, idx] for i, idx in enumerate(keep_idxs)]
        avg_kept  += float(np.mean([k.size(0) for k in kept]))
        n_batches += 1
        for p_text, g in zip(llava.generate(kept, qs, max_new_tokens=max_new_tokens), golds):
            n_correct += vqa_match(p_text, g)
            n_total   += 1
        pbar.set_postfix(acc=f"{n_correct / max(n_total, 1):.3f}")
    return {
        "accuracy":          n_correct / max(n_total, 1),
        "n_correct":         float(n_correct),
        "n_total":           n_total,
        "avg_kept_tokens":   float(avg_kept / max(n_batches, 1)),
    }


# =============================================================================
# TRAINING STEP
# =============================================================================

def train_step(llava: LLaVAWithQGTP, encoders: FrozenEncoders, qgtp: QGTPController,
               batch: List[dict], optimizer,
               grad_clip: float = 1.0,
               grad_accum_steps: int = 1,
               accum_idx: int = 0):
    """One forward/backward pass with optional gradient accumulation."""
    imgs = [b["image"]    for b in batch]
    qs   = [b["question"] for b in batch]
    a_s  = [b["answer"]   for b in batch]
    cp, cpo, ct, dp = encoders.encode(imgs, qs)
    keep_idxs = qgtp.select(cp, cpo, ct, dp)
    kept = [cp[i, idx] for i, idx in enumerate(keep_idxs)]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = llava.forward_loss(kept, qs, a_s) / float(grad_accum_steps)
    loss.backward()
    is_last = (accum_idx + 1) == grad_accum_steps
    if is_last:
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(llava.trainable_parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss.item() * float(grad_accum_steps)


# =============================================================================
# HF UTILITIES
# =============================================================================

def setup_hf():
    """Login to HF Hub (skipped in offline mode)."""
    if OFFLINE:
        print("[hf] OFFLINE mode — using cached files only.")
        return
    if not HF_TOKEN:
        print("[hf] no HF_TOKEN set; public downloads work but uploads will fail.")
        return
    login(token=HF_TOKEN)


def upload_results(name: str, results: dict, local_dir: Optional[str] = None):
    """Save results JSON locally and optionally upload to HF Hub."""
    if local_dir is None:
        local_dir = os.environ.get("RESULTS_DIR", "./results")
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, name)
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[results] wrote {local_path}")
    if OFFLINE or not HF_TOKEN:
        return
    try:
        upload_file(path_or_fileobj=local_path, path_in_repo=name,
                    repo_id=HF_REPO, token=HF_TOKEN)
        print(f"[upload] -> {HF_REPO}/{name}")
    except Exception as e:
        print(f"[upload] FAILED ({e}); local copy remains at {local_path}")


def average_state_dicts(sd_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Equal-weighted FedAvg over a list of state dicts."""
    out  = OrderedDict()
    keys = sd_list[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in sd_list], dim=0)
        out[k]  = stacked.mean(dim=0).to(sd_list[0][k].dtype)
    return out


def cosine_warmup_lr(optimizer, total_steps: int, warmup_frac: float = 0.05,
                     min_lr_ratio: float = 0.1):
    """Cosine LR schedule with linear warm-up."""
    warmup = max(1, int(warmup_frac * total_steps))

    def fn(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        cos  = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
