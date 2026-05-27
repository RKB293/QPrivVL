"""
module_14_attack_grad_leak_v3.py
=================================

Gradient-leakage (iDLG) attack on six federated VQA-RAD checkpoints.
Single file, six settings, merged JSON + figure, HF upload.

Fixes over v2:
  - Per-tensor NORMALIZED L2 gradient matching (scale-invariant, unbounded).
    Cosine was capped at [0, 2] per tensor, which made aggressive-pruning
    settings look easier to attack than 'off'. Normalized L2 stays unbounded.
  - Primary metric is now on the FULL 576x1024 feature (not kept-only).
    Defense goal is to protect the whole image; pruned rows that stay at
    noise are a defense win, not a measurement gap.
  - DLG_STEPS bumped to 1500 so 'off' has a real chance to converge.
  - Per-sample match-loss trajectory logged (every 50 steps) so we can
    verify convergence before trusting the numbers.
"""

import os, json, hashlib, warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import torch, torch.nn.functional as F, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from huggingface_hub import upload_file, hf_hub_download


from qgtp_lib import (setup_hf, FrozenEncoders, load_student_from_hf,
                      QGTPController, LLaVAWithQGTP,
                      HF_REPO, HF_TOKEN, OFFLINE)
from module_10_attack_common import (DEVICE, load_dataset_with_attribute,
                                     free_cuda, gpu_status)

# ----------------------------- CONFIG ---------------------------------------
DATASET_KEY  = "vqarad"
N_TARGETS    = 8
DLG_STEPS    = 1500          # bumped from 300; 'off' needs room to converge
DLG_LR       = 0.1           # lower lr with more steps is more stable
LOG_EVERY    = 50            # match-loss trajectory snapshots
LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.05
SEED         = 42
RESULTS_DIR  = os.environ.get("RESULTS_DIR", "./results")
HF_FOLDER    = "GRADLEAK_Updated"
os.makedirs(RESULTS_DIR, exist_ok=True)

SETTINGS = [
    ("off",          "lora_federated_vqarad_off.pt",          "off",     None),
    ("fixed_rho0.3", "lora_federated_vqarad_fixed_rho0.3.pt", "fixed",   0.3),
    ("fixed_rho0.5", "lora_federated_vqarad_fixed_rho0.5.pt", "fixed",   0.5),
    ("fixed_rho0.7", "lora_federated_vqarad_fixed_rho0.7.pt", "fixed",   0.7),
    ("fixed_rho0.9", "lora_federated_vqarad_fixed_rho0.9.pt", "fixed",   0.9),
    ("student",      "lora_federated_vqarad_student.pt",      "student", None),
]

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(SEED); np.random.seed(SEED)
setup_hf()
print(f"[gpu] {gpu_status()}")

# ----------------------------- DATA + ENCODERS + STUDENT --------------------
print("[data] loading VQA-RAD")
samples, _ = load_dataset_with_attribute(DATASET_KEY, N_TARGETS + 50)
TARGET_IDX = list(range(N_TARGETS))

print("[model] loading frozen encoders + student")
encoders = FrozenEncoders()
student  = load_student_from_hf()

# ----------------------------- LORA LOADER (HARD ASSERT) --------------------
def load_lora_with_check(llava, lora_filename):
    local = hf_hub_download(repo_id=HF_REPO, filename=f"lora/{lora_filename}",
                            token=HF_TOKEN)
    assert os.path.exists(local), f"LoRA file missing: {local}"
    with open(local, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    blob = torch.load(local, map_location="cpu")
    sd   = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    llava.load_lora_state_dict(sd)
    for n, p in llava.llava.named_parameters():
        if "lora_" in n: p.data = p.data.float()
    print(f"   [lora] {lora_filename}  sha256[:16]={sha}")
    return sha

# ----------------------------- ATTACK CORE ----------------------------------
def normalized_l2_match(dummy_grads, real_grads):
    """Per-tensor scale-invariant L2. Unbounded above, 0 at perfect match.
    sum_t ||dg_t - rg_t||^2 / (||rg_t||^2 + eps)."""
    total = 0.0
    for dg, rg in zip(dummy_grads, real_grads):
        dgf, rgf = dg.float(), rg.float()
        denom = (rgf ** 2).sum().clamp(min=1e-8)
        total = total + ((dgf - rgf) ** 2).sum() / denom
    return total

def feature_psnr(a, b):
    mse = F.mse_loss(a, b).clamp(min=1e-12)
    return float(-10.0 * torch.log10(mse))

def capture_real(sample_idx, qgtp, llava, lora_params):
    s = samples[sample_idx]
    img, q, a = s["image"], s["question"], s["answer"]
    cp, cpo, ct, dp = encoders.encode([img], [q])
    keep_idx = qgtp.select(cp, cpo, ct, dp)[0]
    kept = [cp[0, keep_idx].to(torch.bfloat16)]
    for p in lora_params: p.grad = None
    loss = llava.forward_loss(kept, [q], [a])
    loss.backward()
    real_grads  = [p.grad.detach().clone() for p in lora_params]
    real_visual = cp[0].detach().float().clone()                  # (576, 1024)
    return real_grads, real_visual, keep_idx, q, a

def invert_idlg(real_grads, real_visual, keep_idx, q, a,
                llava, lora_params, steps=DLG_STEPS):
    """Optimize dummy (576,1024). Defender's keep_idx is applied as a fixed
    gather (attacker doesn't re-prune). Returns best_dummy, trajectory."""
    dummy = torch.randn_like(real_visual).unsqueeze(0)
    dummy.requires_grad_(True)
    opt = torch.optim.Adam([dummy], lr=DLG_LR)
    best_loss, best_dummy = float("inf"), None
    traj = []
    for step in range(steps):
        kept_dummy = [dummy[0, keep_idx].to(torch.bfloat16)]
        for p in lora_params: p.grad = None
        loss_lm = llava.forward_loss(kept_dummy, [q], [a])
        dgrads  = torch.autograd.grad(loss_lm, lora_params,
                                      create_graph=True, retain_graph=True)
        m_loss = normalized_l2_match(dgrads, real_grads)
        tv = ((dummy[:, 1:] - dummy[:, :-1]) ** 2).mean()
        total = m_loss + 1e-4 * tv
        opt.zero_grad(); total.backward(); opt.step()
        if step % LOG_EVERY == 0:
            traj.append({"step": step, "match_loss": float(m_loss.item())})
        if m_loss.item() < best_loss:
            best_loss  = float(m_loss.item())
            best_dummy = dummy.detach().clone()
    return best_dummy.squeeze(0), best_loss, traj

# ----------------------------- MAIN LOOP ------------------------------------
all_results = {}
N_TOKENS    = 576

print(f"\n[run] iDLG (norm-L2), {DLG_STEPS} steps, lr={DLG_LR}")
for name, fname, mode, rho in SETTINGS:
    print(f"\n--- {name}  (mode={mode}, rho={rho}) ---")
    llava = LLaVAWithQGTP(lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                          lora_dropout=LORA_DROPOUT)
    llava.llava.eval()
    sha = load_lora_with_check(llava, fname)
    lora_params = [p for n, p in llava.llava.named_parameters() if "lora_" in n]
    qgtp = QGTPController(mode=mode, fixed_rho=rho or 0.5,
                          student=student if mode != "off" else None)

    per_sample = []
    for ti, idx in enumerate(TARGET_IDX):
        real_grads, real_visual, keep_idx, q, a = capture_real(
            idx, qgtp, llava, lora_params)
        dummy, mloss, traj = invert_idlg(real_grads, real_visual, keep_idx,
                                         q, a, llava, lora_params)
        # FULL-tensor metrics (primary): dummy includes un-optimized noise
        # rows on pruned positions -- that's exactly what we want to measure.
        psnr_full = feature_psnr(dummy, real_visual)
        cos_full  = F.cosine_similarity(dummy.flatten(),
                                        real_visual.flatten(), dim=0).item()
        # Kept-rows metrics (secondary)
        rk, dk = real_visual[keep_idx], dummy[keep_idx]
        psnr_kept = feature_psnr(dk, rk)
        cos_kept  = F.cosine_similarity(dk.flatten(), rk.flatten(), dim=0).item()
        n_kept    = int(keep_idx.numel())
        cur_rho   = 1.0 - n_kept / N_TOKENS
        per_sample.append({
            "sample_idx": int(idx), "n_kept": n_kept, "rho": float(cur_rho),
            "match_loss_final": float(mloss),
            "match_loss_start": float(traj[0]["match_loss"]) if traj else None,
            "psnr_full": float(psnr_full), "cos_full": float(cos_full),
            "psnr_kept": float(psnr_kept), "cos_kept": float(cos_kept),
            "trajectory": traj,
        })
        print(f"   [{ti+1}/{N_TARGETS}] n_kept={n_kept:3d}/576  "
              f"mloss {traj[0]['match_loss']:7.1f}->{mloss:7.1f}  "
              f"psnr_full={psnr_full:.2f}  cos_full={cos_full:.3f}  "
              f"cos_kept={cos_kept:.3f}")
        del real_grads, real_visual, dummy; free_cuda()

    pf  = [r["psnr_full"] for r in per_sample]
    cf  = [r["cos_full"]  for r in per_sample]
    pk  = [r["psnr_kept"] for r in per_sample]
    ck  = [r["cos_kept"]  for r in per_sample]
    rhos = [r["rho"]      for r in per_sample]
    ml_start = [r["match_loss_start"] for r in per_sample]
    ml_end   = [r["match_loss_final"] for r in per_sample]
    all_results[name] = {
        "qgtp_mode": mode, "fixed_rho": rho, "lora_sha256": sha,
        "lora_file": fname,
        "rho_mean": float(np.mean(rhos)),
        "n_kept_mean": float(np.mean([r["n_kept"] for r in per_sample])),
        "match_loss_start_mean": float(np.mean(ml_start)),
        "match_loss_final_mean": float(np.mean(ml_end)),
        "psnr_full_mean": float(np.mean(pf)), "psnr_full_std": float(np.std(pf)),
        "cos_full_mean":  float(np.mean(cf)), "cos_full_std":  float(np.std(cf)),
        "psnr_kept_mean": float(np.mean(pk)), "psnr_kept_std": float(np.std(pk)),
        "cos_kept_mean":  float(np.mean(ck)), "cos_kept_std":  float(np.std(ck)),
        "per_sample": per_sample,
    }
    print(f"   >> mloss {np.mean(ml_start):.1f} -> {np.mean(ml_end):.1f}   "
          f"psnr_full={np.mean(pf):.2f}+/-{np.std(pf):.2f}   "
          f"cos_full={np.mean(cf):.3f}+/-{np.std(cf):.3f}")
    del llava, lora_params, qgtp; free_cuda()

del encoders, student; free_cuda()
print(f"\n[free] {gpu_status()}")

# ----------------------------- SAVE JSON ------------------------------------
order = ["off", "fixed_rho0.3", "fixed_rho0.5", "fixed_rho0.7",
         "fixed_rho0.9", "student"]
blob = {
    "attack": "GRAD_LEAK_v3", "variant": "iDLG_normL2",
    "dataset": DATASET_KEY, "n_targets": N_TARGETS,
    "dlg_steps": DLG_STEPS, "dlg_lr": DLG_LR,
    "metric_note": ("PRIMARY: cos_full / psnr_full on full 576x1024 feature. "
                    "SECONDARY: cos_kept / psnr_kept on defender's kept rows. "
                    "Match loss is per-tensor normalized L2, sum over LoRA tensors. "
                    "Lower cos/psnr = better defense."),
    "settings_order": order,
    "results": {k: all_results[k] for k in order if k in all_results},
}
json_path = os.path.join(RESULTS_DIR, "results_grad_leak_federated_vqarad_all.json")
with open(json_path, "w") as f: json.dump(blob, f, indent=2, default=str)
print(f"[json] wrote {json_path}")

# ----------------------------- FIGURES --------------------------------------
names = [k for k in order if k in all_results]
psnr_m = [all_results[k]["psnr_full_mean"] for k in names]
psnr_s = [all_results[k]["psnr_full_std"]  for k in names]
cos_m  = [all_results[k]["cos_full_mean"]  for k in names]
cos_s  = [all_results[k]["cos_full_std"]   for k in names]

# (A) Merged bar chart
fig, ax = plt.subplots(figsize=(9, 4.5))
ax2 = ax.twinx()
x = np.arange(len(names)); w = 0.35
b1 = ax.bar(x - w/2, psnr_m, w, yerr=psnr_s, color="#d62728",
            label="PSNR full (dB)", capsize=3, alpha=0.85)
b2 = ax2.bar(x + w/2, cos_m, w, yerr=cos_s, color="#1f77b4",
             label="cos sim full", capsize=3, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
ax.set_ylabel("PSNR full (dB)  -- lower = better defense", color="#d62728")
ax2.set_ylabel("Cos sim full -- lower = better defense", color="#1f77b4")
ax.grid(alpha=0.3, axis="y")
ax.set_title("iDLG (norm-L2) on federated VQA-RAD LoRA gradients\n"
             "metrics on full 576x1024 feature, n=8 samples")
ax.legend([b1, b2], ["PSNR full (dB)", "cos sim full"], loc="upper left")
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "fig_grad_leak_federated_vqarad_all.pdf")
plt.savefig(fig_path); plt.close()
print(f"[fig ] wrote {fig_path}")

# (B) Convergence diagnostic
fig2, ax = plt.subplots(figsize=(9, 4.5))
for name in names:
    traj = all_results[name]["per_sample"][0]["trajectory"]  # sample 0 traj
    steps = [t["step"] for t in traj]
    vals  = [t["match_loss"] for t in traj]
    ax.plot(steps, vals, marker="o", markersize=3, label=name, alpha=0.85)
ax.set_xlabel("DLG step"); ax.set_ylabel("match loss (normalized L2)")
ax.set_yscale("log")
ax.set_title("iDLG match-loss trajectory (sample 0 per setting)")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
fig_traj_path = os.path.join(RESULTS_DIR, "fig_grad_leak_convergence.pdf")
plt.savefig(fig_traj_path); plt.close()
print(f"[fig ] wrote {fig_traj_path}")

# ----------------------------- UPLOAD ---------------------------------------
def _upload(local, remote):
    try:
        upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=HF_REPO, token=HF_TOKEN)
        print(f"  uploaded -> {HF_REPO}/{remote}")
    except Exception as e:
        print(f"  FAILED ({remote}): {e}")

if OFFLINE or not HF_TOKEN:
    print("[upload] skipping (offline/no token)")
else:
    print(f"[upload] -> {HF_REPO}/{HF_FOLDER}/")
    _upload(json_path,     f"{HF_FOLDER}/results_grad_leak_federated_vqarad_all.json")
    _upload(fig_path,      f"{HF_FOLDER}/fig_grad_leak_federated_vqarad_all.pdf")
    _upload(fig_traj_path, f"{HF_FOLDER}/fig_grad_leak_convergence.pdf")

print(f"\n[done] {gpu_status()}")
