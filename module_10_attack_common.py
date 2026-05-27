import os
import gc
import json
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
from tqdm.auto import tqdm

from huggingface_hub import hf_hub_download

from qgtp_lib import (
    HF_REPO, HF_TOKEN,
    FrozenEncoders, load_student_from_hf,
    QGTPController, LLaVAWithQGTP,
    load_dataset_split,
    _get_inner,
)

# =============================================================================
# CONFIG (overridable per attack module)
# =============================================================================
DEVICE        = "cuda"
DATASET_KEY   = "vqarad"
CUT_LAYER     = 16          
PIXEL_SIZE    = 112         
PUBLIC_SIZE   = 800
PRIVATE_SIZE  = 400


# =============================================================================
# MODEL TAG RESOLUTION
# =============================================================================
def lora_repo_path(model_tag: str) -> str:
    return f"lora/lora_{model_tag}.pt"


def load_lora_for_tag(llava: LLaVAWithQGTP, model_tag: str,
                       results_dir: str = "./results") -> bool:
    """Try local results dir, then HF Hub. Returns True if loaded."""
    # 1) Local file
    local_paths = [
        os.path.join(results_dir, f"lora_{model_tag}.pt"),
        os.path.join("./results", f"lora_{model_tag}.pt"),
    ]
    for p in local_paths:
        if os.path.exists(p):
            print(f"[lora] loading local: {p}")
            sd = torch.load(p, map_location="cpu")
            llava.load_lora_state_dict(sd["state_dict"])
            return True
    # 2) HF Hub
    try:
        path = hf_hub_download(repo_id=HF_REPO,
                               filename=lora_repo_path(model_tag),
                               token=HF_TOKEN)
        print(f"[lora] downloaded from HF: {path}")
        sd = torch.load(path, map_location="cpu")
        llava.load_lora_state_dict(sd["state_dict"])
        return True
    except Exception as e:
        print(f"[lora] could not load LoRA for tag '{model_tag}': {e}")
        print(f"[lora] proceeding with freshly initialized LoRA (results "
              f"will be noisier; expect lower attack saturation values)")
        return False


# =============================================================================
# DATA
# =============================================================================
def parse_dataset_from_tag(model_tag: str) -> str:
    """direct_vqarad_off -> 'vqarad', split_gqa_fixed_cut16_rho0.5 -> 'gqa', etc."""
    DATASETS = ("vqarad", "vqav2", "gqa", "okvqa", "slake", "pathvqa", "rsvqa")
    parts = model_tag.split("_")
    for part in parts:
        if part in DATASETS:
            return part
    raise ValueError(f"could not parse dataset from MODEL_TAG: {model_tag}")


def load_dataset_with_attribute(dataset_key: str, n_max: int):
    """Returns (samples, attribute_labels). attribute_labels is a list of
    strings used as the binary MIA target.
    
    For VQA-RAD: re-fetched answer_type ('OPEN' / 'CLOSED' / 'OTHER').
    For others: derived from the answer string ('YESNO' if yes/no, else 'OPEN').
    """
    samples = load_dataset_split(dataset_key, max_samples=n_max)
    
    if dataset_key == "vqarad":
        from datasets import load_dataset
        raw = load_dataset("flaviagiammarino/vqa-rad", split="train")
        attrs = []
        for i in range(len(samples)):
            at = str(raw[i].get("answer_type", "")).upper()
            attrs.append(at if at in ("OPEN", "CLOSED") else "OTHER")
        del raw
    else:
        # Generic fallback: yes/no answers vs everything else.
        attrs = []
        for s in samples:
            ans = str(s.get("answer", "")).strip().lower()
            attrs.append("CLOSED" if ans in ("yes", "no") else "OPEN")
    return samples, attrs


def public_private_split(n_total: int, public: int, private: int,
                          seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_total)
    n = min(public + private, n_total)
    return order[:public].tolist(), order[public:n].tolist()


# =============================================================================
# SMASHED-DATA EXTRACTOR
# =============================================================================
class SmashedExtractor:
    """Captures the hidden state arriving INTO CUT_LAYER of LLaVA's LM,
    under a chosen pruning policy. Cached to disk per (model_tag,
    qgtp_mode, fixed_rho)."""

    def __init__(self, llava: LLaVAWithQGTP, encoders: FrozenEncoders,
                 student=None, cut_layer: int = CUT_LAYER):
        self.llava    = llava
        self.encoders = encoders
        self.student  = student
        self.cut_layer = cut_layer

        lm = _get_inner(llava.llava, "language_model")
        layers = lm.layers
        assert 0 < cut_layer < len(layers), (
            f"CUT_LAYER {cut_layer} out of range 0..{len(layers)-1}")
        self._capture: Dict[str, Any] = {"h": None}

        def _hook(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            self._capture["h"] = h.detach().to(torch.float32).cpu()
            return None

        self._handle = layers[cut_layer].register_forward_pre_hook(
            _hook, with_kwargs=True)

    def close(self):
        if hasattr(self, "_handle") and self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @torch.no_grad()
    def extract_for_samples(self, samples_list: List[dict],
                            indices: List[int],
                            qgtp_mode: str,
                            fixed_rho: Optional[float] = None,
                            batch_size: int = 8,
                            desc: str = "extract") -> Dict[str, Any]:
        """For each sample i in `indices`, run forward up to CUT_LAYER under
        the pruning policy and return per-sample smashed activations."""
        qgtp = QGTPController(
            mode=qgtp_mode,
            fixed_rho=fixed_rho or 0.5,
            student=self.student if qgtp_mode == "student" else None,
        )
        smashed_list, rhos, n_kept_list, attn_lens = [], [], [], []
        pbar = tqdm(range(0, len(indices), batch_size), desc=desc, leave=False)
        for s in pbar:
            chunk = indices[s:s + batch_size]
            batch = [samples_list[i] for i in chunk]
            imgs  = [b["image"]    for b in batch]
            qs    = [b["question"] for b in batch]
            cp, cpo, ct, dp = self.encoders.encode(imgs, qs)
            keep_idxs = qgtp.select(cp, cpo, ct, dp)
            kept = [cp[i, idx] for i, idx in enumerate(keep_idxs)]

            embeds, attn, _ = self.llava._build_embeds(kept, qs)
            _ = self.llava.llava(inputs_embeds=embeds, attention_mask=attn,
                                  use_cache=False)
            h = self._capture["h"]                       # (B, L, D) on CPU
            attn_cpu = attn.cpu()
            for b in range(h.size(0)):
                # _build_embeds left-pads in generation layout, so the real
                # tokens are at the END of the sequence.
                real_len = int(attn_cpu[b].sum().item())
                smashed_list.append(h[b, -real_len:].clone())
                n_kept_list.append(int(kept[b].size(0)))
                attn_lens.append(real_len)
                if qgtp_mode == "off":
                    rhos.append(0.0)
                elif qgtp_mode == "fixed":
                    rhos.append(float(fixed_rho))
                else:
                    rhos.append(1.0 - kept[b].size(0) / 576.0)

        return {
            "smashed":   smashed_list,
            "rhos":      np.asarray(rhos, dtype=np.float32),
            "n_kept":    np.asarray(n_kept_list, dtype=np.int32),
            "attn_lens": np.asarray(attn_lens, dtype=np.int32),
        }


def smashed_cache_path(cache_dir: str, model_tag: str,
                        qgtp_mode: str, fixed_rho: Optional[float]) -> str:
    if qgtp_mode == "fixed":
        suffix = f"fixed{fixed_rho}"
    else:
        suffix = qgtp_mode
    return os.path.join(cache_dir, f"smashed_{model_tag}_{suffix}.pt")


def get_or_extract_smashed(extractor: SmashedExtractor,
                            samples_list: List[dict],
                            public_idx: List[int], private_idx: List[int],
                            cache_dir: str, model_tag: str,
                            qgtp_mode: str,
                            fixed_rho: Optional[float] = None) -> Dict[str, Any]:
    """Idempotent: returns cached extraction or runs it and caches."""
    cp = smashed_cache_path(cache_dir, model_tag, qgtp_mode, fixed_rho)
    if os.path.exists(cp):
        print(f"[cache] loading {cp}")
        return torch.load(cp, weights_only=False)

    print(f"[extract] {qgtp_mode}  fixed_rho={fixed_rho}  tag={model_tag}")
    pub  = extractor.extract_for_samples(samples_list, public_idx, qgtp_mode,
                                          fixed_rho, desc=f"{qgtp_mode} pub")
    priv = extractor.extract_for_samples(samples_list, private_idx, qgtp_mode,
                                          fixed_rho, desc=f"{qgtp_mode} priv")
    out = {"public": pub, "private": priv}
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(out, cp)
    print(f"[cache] saved {cp}")
    return out


# =============================================================================
# ATTACK MODELS
# =============================================================================
class FeatureDecoder(nn.Module):
    """Smashed sequence -> CLIP-V patch grid (576, 1024).

    Cross-attention with 576 learnable queries pools the variable-length
    smashed sequence into a fixed-size grid. Two cross-attn + FFN blocks.
    """
    def __init__(self, in_dim: int = 4096, out_tokens: int = 576,
                 out_dim: int = 1024, hidden: int = 512,
                 n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(out_tokens, hidden) * 0.02)
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers   = nn.ModuleList([
            nn.MultiheadAttention(hidden, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norms_q = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                          nn.Linear(hidden * 2, hidden))
            for _ in range(n_layers)
        ])
        self.norms_f = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, smashed_padded, key_padding_mask):
        kv = self.in_proj(smashed_padded)
        q  = self.queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        for attn, lnq, ff, lnf in zip(self.layers, self.norms_q,
                                       self.ffns, self.norms_f):
            q2, _ = attn(q, kv, kv, key_padding_mask=key_padding_mask,
                         need_weights=False)
            q = lnq(q + q2)
            q = lnf(q + ff(q))
        return self.out_proj(q)


class PixelDecoder(nn.Module):
    """Smashed sequence -> (3, PIXEL_SIZE, PIXEL_SIZE) image."""
    def __init__(self, in_dim: int = 4096, hidden: int = 384,
                 n_heads: int = 6, out_size: int = PIXEL_SIZE):
        super().__init__()
        self.query   = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.in_proj = nn.Linear(in_dim, hidden)
        self.attn    = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm    = nn.LayerNorm(hidden)
        self.fc      = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, 256 * 7 * 7),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 4, 2, 1), nn.GELU(), nn.BatchNorm2d(192),
            nn.ConvTranspose2d(192, 128, 4, 2, 1), nn.GELU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.GELU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,  32, 4, 2, 1),  nn.GELU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.out_size = out_size

    def forward(self, smashed_padded, key_padding_mask):
        kv = self.in_proj(smashed_padded)
        q  = self.query.unsqueeze(0).expand(kv.size(0), -1, -1)
        q2, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                          need_weights=False)
        q = self.norm(q + q2).squeeze(1)
        x = self.fc(q).view(-1, 256, 7, 7)
        img = self.up(x)
        if self.out_size != 112:
            img = F.interpolate(img, size=(self.out_size, self.out_size),
                                mode="bilinear", align_corners=False)
        return img


class MIAClassifier(nn.Module):
    """Smashed sequence -> binary attribute (e.g. OPEN/CLOSED)."""
    def __init__(self, in_dim: int = 4096, hidden: int = 256):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.attn    = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.head    = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, smashed_padded, key_padding_mask):
        kv = self.in_proj(smashed_padded)
        q  = kv.mean(dim=1, keepdim=True)
        q2, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                          need_weights=False)
        return self.head(q2.squeeze(1)).squeeze(-1)


# =============================================================================
# COMMON UTILITIES
# =============================================================================
def pad_batch(smashed_list):
    """Right-pad list of (T_i, D) -> (B, L_max, D) and key_padding_mask
    where True = PAD."""
    L_max = max(t.size(0) for t in smashed_list)
    D     = smashed_list[0].size(-1)
    B     = len(smashed_list)
    out  = torch.zeros(B, L_max, D, dtype=torch.float32)
    mask = torch.ones(B, L_max, dtype=torch.bool)
    for i, t in enumerate(smashed_list):
        L = t.size(0)
        out[i, :L]   = t
        mask[i, :L]  = False
    return out, mask


def get_clip_patches_for_idx(encoders: FrozenEncoders, samples_list: List[dict],
                              indices: List[int], chunk: int = 16):
    """CLIP-V patch grid (576, 1024) per sample."""
    out = []
    for s in range(0, len(indices), chunk):
        imgs = [samples_list[i]["image"] for i in indices[s:s + chunk]]
        cx = encoders.clip_v_p(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        cout = encoders.clip_v(pixel_values=cx, output_hidden_states=True)
        out.append(cout.hidden_states[-2][:, 1:].cpu().float())
    return torch.cat(out, dim=0)


def get_pixel_targets_for_idx(samples_list: List[dict],
                               indices: List[int],
                               size: int = PIXEL_SIZE):
    out = []
    for i in indices:
        img = samples_list[i]["image"].convert("RGB").resize(
            (size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        out.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(out)


# Defense-sweep settings used by FSHA, FORA, MIA.
DEFAULT_RHO_SWEEP = [0.0, 0.3, 0.5, 0.7, 0.9]


def build_settings(rho_sweep=None, include_student=True):
    rho_sweep = rho_sweep if rho_sweep is not None else DEFAULT_RHO_SWEEP
    out = [("off", None)]
    for r in rho_sweep:
        if r == 0.0: continue
        out.append(("fixed", float(r)))
    if include_student:
        out.append(("student", None))
    return out


def setting_key(qgtp_mode, fixed_rho):
    return f"fixed_{fixed_rho}" if qgtp_mode == "fixed" else qgtp_mode


def setting_label(qgtp_mode, fixed_rho):
    if qgtp_mode == "off":     return r"off ($\rho$=0)"
    if qgtp_mode == "student": return r"student (learned $\rho$)"
    return rf"fixed ($\rho$={fixed_rho})"


# Free GPU helpers
def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_status():
    if not torch.cuda.is_available():
        return "no CUDA"
    free, total = torch.cuda.mem_get_info(0)
    return f"GPU 0: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB"

