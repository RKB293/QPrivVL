import os, gc, io, json, tempfile, warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, hf_hub_download, upload_file


from qgtp_lib import (
    setup_hf, FrozenEncoders, load_student_from_hf, LLaVAWithQGTP,
    load_dataset_split, HF_REPO, HF_TOKEN,
)
from module_10_attack_common import (
    DEVICE, CUT_LAYER, SmashedExtractor, MIAClassifier, pad_batch, free_cuda,
)

# ============================================================================
# CONFIG
# ============================================================================
PUBLIC_SIZE, PRIVATE_SIZE = 800, 400
EPOCHS, BATCH, LR, SEED = 6, 16, 1e-4, 42
LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.05
CACHE_DIR_LOCAL = "./mia_cache"
os.makedirs(CACHE_DIR_LOCAL, exist_ok=True)

REMOTE_CACHE = "updated_MIA/cache"
REMOTE_FINAL = "updated_MIA"

# The 24 checkpoints. (paradigm, defense_key, lora_filename, qgtp_mode, fixed_rho)
DEFENSES = [
    ("off",         "off",     None),
    ("fixed_rho0.3","fixed",   0.3),
    ("fixed_rho0.5","fixed",   0.5),
    ("fixed_rho0.7","fixed",   0.7),
    ("fixed_rho0.9","fixed",   0.9),
    ("student",     "student", None),
]
def _lora_name(paradigm, defense):
    if paradigm == "direct":
        return f"lora/lora_direct_vqarad_{defense}.pt"
    if paradigm == "federated":
        return f"lora/lora_federated_vqarad_{defense}.pt"
    if paradigm == "split":
        m = {"off":"off_cut16","student":"student_cut16",
             "fixed_rho0.3":"fixed_cut16_rho0.3","fixed_rho0.5":"fixed_cut16_rho0.5",
             "fixed_rho0.7":"fixed_cut16_rho0.7","fixed_rho0.9":"fixed_cut16_rho0.9"}
        return f"lora/lora_split_vqarad_{m[defense]}.pt"
    if paradigm == "ushape":
        m = {"off":"off_a8b24","student":"student_a8b24",
             "fixed_rho0.3":"fixed_a8b24_rho0.3","fixed_rho0.5":"fixed_a8b24_rho0.5",
             "fixed_rho0.7":"fixed_a8b24_rho0.7","fixed_rho0.9":"fixed_a8b24_rho0.9"}
        return f"lora/lora_ushape_vqarad_{m[defense]}.pt"
    raise ValueError(paradigm)

JOBS = []
for paradigm in ("direct","federated","split","ushape"):
    for defense, qgtp_mode, fixed_rho in DEFENSES:
        JOBS.append({
            "paradigm": paradigm, "defense": defense,
            "lora_file": _lora_name(paradigm, defense),
            "qgtp_mode": qgtp_mode, "fixed_rho": fixed_rho,
            "result_name": f"result_{paradigm}_{defense}.json",
        })

# ============================================================================
# HF I/O
# ============================================================================
setup_hf()
api = HfApi(token=HF_TOKEN)

def _upload(local_path, remote_path):
    upload_file(path_or_fileobj=local_path, path_in_repo=remote_path,
                repo_id=HF_REPO, token=HF_TOKEN)

def _upload_bytes(data_bytes, remote_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
        tf.write(data_bytes); tmp = tf.name
    _upload(tmp, remote_path); os.unlink(tmp)

def _list_cache():
    try:
        files = api.list_repo_files(repo_id=HF_REPO)
        return {os.path.basename(f) for f in files if f.startswith(REMOTE_CACHE + "/")}
    except Exception:
        return set()

def _download_cache(name):
    p = hf_hub_download(repo_id=HF_REPO, filename=f"{REMOTE_CACHE}/{name}", token=HF_TOKEN)
    with open(p) as f: return json.load(f)

# ============================================================================
# LABELS (VQA-RAD)
# ============================================================================
def vqarad_labels(samples):
    """yes/no answer -> CLOSED=0, otherwise OPEN=1."""
    out = []
    for s in samples:
        ans = str(s.get("answer","")).strip().lower()
        out.append(0 if ans in ("yes","no") else 1)
    return out

# ============================================================================
# MIA TRAIN / EVAL
# ============================================================================
def fit_mia(sm_pub, y_pub):
    clf = MIAClassifier().to(DEVICE).train()
    opt = torch.optim.AdamW(clf.parameters(), lr=LR)
    n = len(sm_pub)
    for ep in range(EPOCHS):
        perm = np.random.permutation(n)
        for s in tqdm(range(0,n,BATCH), desc=f"fit ep{ep+1}", leave=False):
            ids = perm[s:s+BATCH].tolist()
            sm = [sm_pub[i] for i in ids]
            x, m = pad_batch(sm)
            y = torch.tensor([y_pub[i] for i in ids], dtype=torch.float32, device=DEVICE)
            logits = clf(x.to(DEVICE), m.to(DEVICE))
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval(); return clf

@torch.no_grad()
def eval_mia(clf, sm_priv, y_priv):
    probs = []
    for s in range(0,len(sm_priv),BATCH):
        x, m = pad_batch(sm_priv[s:s+BATCH])
        p = torch.sigmoid(clf(x.to(DEVICE), m.to(DEVICE))).cpu().numpy()
        probs.extend(p.tolist())
    probs = np.asarray(probs); ys = np.asarray(y_priv)
    return {
        "auc": float(roc_auc_score(ys, probs)),
        "acc": float(((probs>0.5).astype(int)==ys).mean()),
    }

# ============================================================================
# RUN ONE JOB
# ============================================================================
def run_one(job, samples, public_idx, private_idx, y_pub, y_priv):
    print(f"\n=== {job['paradigm']} / {job['defense']} ===")
    # 1) Load LoRA -- FAIL LOUDLY
    lora_path = hf_hub_download(repo_id=HF_REPO, filename=job["lora_file"], token=HF_TOKEN)
    sd = torch.load(lora_path, map_location="cpu")

    encoders = FrozenEncoders()
    student  = load_student_from_hf() if job["qgtp_mode"] == "student" else None
    llava    = LLaVAWithQGTP(lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    llava.load_lora_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    llava.llava.eval()

    # 2) Extract smashed under matching defense
    extr = SmashedExtractor(llava, encoders, student=student, cut_layer=CUT_LAYER)
    pub  = extr.extract_for_samples(samples, public_idx,
                                     job["qgtp_mode"], job["fixed_rho"], desc="pub")
    priv = extr.extract_for_samples(samples, private_idx,
                                     job["qgtp_mode"], job["fixed_rho"], desc="priv")
    extr.close()
    del encoders, student, llava; free_cuda()

    # 3) Train + eval attacker
    clf = fit_mia(pub["smashed"], y_pub)
    metrics = eval_mia(clf, priv["smashed"], y_priv)
    del clf; free_cuda()

    return {
        "paradigm": job["paradigm"],
        "defense":  job["defense"],
        "qgtp_mode": job["qgtp_mode"],
        "fixed_rho": job["fixed_rho"],
        "rho_mean": float(np.mean(priv["rhos"])),
        "rho_std":  float(np.std(priv["rhos"])),
        "n_kept_mean": float(np.mean(priv["n_kept"])),
        "auc": metrics["auc"],
        "acc": metrics["acc"],
    }

# ============================================================================
# MAIN
# ============================================================================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("[data] loading VQA-RAD")
    samples = load_dataset_split("vqarad", max_samples=PUBLIC_SIZE+PRIVATE_SIZE+200)
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(samples))
    public_idx  = order[:PUBLIC_SIZE].tolist()
    private_idx = order[PUBLIC_SIZE:PUBLIC_SIZE+PRIVATE_SIZE].tolist()

    labels = vqarad_labels(samples)
    y_pub  = [labels[i] for i in public_idx]
    y_priv = [labels[i] for i in private_idx]
    n_open_pub, n_open_priv = sum(y_pub), sum(y_priv)
    print(f"[labels] OPEN pub={n_open_pub}/{len(y_pub)}  priv={n_open_priv}/{len(y_priv)}")
    assert 0 < n_open_pub  < len(y_pub),  "public split is one-class -- abort"
    assert 0 < n_open_priv < len(y_priv), "private split is one-class -- abort"

    # Resume: skip jobs whose result is already on HF
    done = _list_cache()
    print(f"[resume] {len(done)} cached results found on HF")

    all_results = []
    # Load any cached results we already have
    for job in JOBS:
        if job["result_name"] in done:
            try:
                all_results.append(_download_cache(job["result_name"]))
                print(f"[skip ] {job['paradigm']}/{job['defense']} -- cached")
            except Exception as e:
                print(f"[warn ] could not read cache {job['result_name']}: {e}")

    done_keys = {(r["paradigm"], r["defense"]) for r in all_results}

    # Run remaining
    for job in JOBS:
        if (job["paradigm"], job["defense"]) in done_keys: continue
        try:
            r = run_one(job, samples, public_idx, private_idx, y_pub, y_priv)
            r["n_open_pub"] = n_open_pub; r["n_open_priv"] = n_open_priv
            all_results.append(r)
            # Upload partial result IMMEDIATELY
            _upload_bytes(json.dumps(r, indent=2).encode(),
                          f"{REMOTE_CACHE}/{job['result_name']}")
            print(f"[ok   ] {job['paradigm']}/{job['defense']} AUC={r['auc']:.3f} acc={r['acc']:.3f}")
        except Exception as e:
            print(f"[FAIL ] {job['paradigm']}/{job['defense']}: {e}")
            free_cuda()

    if len(all_results) < len(JOBS):
        print(f"\n[partial] {len(all_results)}/{len(JOBS)} done. Re-run to resume.")
        return

    # ========================================================================
    # AGGREGATE + FIGURE
    # ========================================================================
    print(f"\n[final] all {len(JOBS)} done -- writing aggregate results")

    final_blob = {
        "attack": "MIA",
        "dataset": "vqarad",
        "cut_layer": CUT_LAYER,
        "attribute": "answer is yes/no (CLOSED=0) vs other (OPEN=1)",
        "n_public": PUBLIC_SIZE, "n_private": PRIVATE_SIZE,
        "n_open_pub": n_open_pub, "n_open_priv": n_open_priv,
        "config": {"epochs":EPOCHS,"batch":BATCH,"lr":LR,"seed":SEED,
                   "lora_r":LORA_R,"lora_alpha":LORA_ALPHA,"lora_dropout":LORA_DROPOUT},
        "results": all_results,
    }
    final_local = "results_mia_vqarad_all24.json"
    with open(final_local,"w") as f: json.dump(final_blob, f, indent=2)
    _upload(final_local, f"{REMOTE_FINAL}/{final_local}")
    print(f"[upload] {REMOTE_FINAL}/{final_local}")

    # Figure
    defense_order = ["off","fixed_rho0.3","fixed_rho0.5","fixed_rho0.7","fixed_rho0.9","student"]
    xt_labels     = ["off","ρ=0.3","ρ=0.5","ρ=0.7","ρ=0.9","student"]
    paradigms     = ["direct","federated","split","ushape"]
    colors        = {"direct":"#1f77b4","federated":"#2ca02c",
                     "split":"#d62728","ushape":"#9467bd"}

    by = {(r["paradigm"], r["defense"]): r for r in all_results}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, title in [(axes[0],"auc","ROC-AUC"),(axes[1],"acc","Accuracy")]:
        for p in paradigms:
            ys = [by[(p,d)][metric] for d in defense_order]
            ax.plot(range(len(defense_order)), ys, marker="o", color=colors[p], label=p)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="random")
        ax.set_xticks(range(len(defense_order))); ax.set_xticklabels(xt_labels, rotation=20)
        ax.set_ylabel(title); ax.set_ylim(0.3, 1.05); ax.grid(alpha=0.3)
        ax.set_title(f"MIA {title} vs defense (VQA-RAD)")
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig_pdf, fig_png = "fig_mia_summary_vqarad.pdf", "fig_mia_summary_vqarad.png"
    plt.savefig(fig_pdf); plt.savefig(fig_png, dpi=150); plt.close()
    _upload(fig_pdf, f"{REMOTE_FINAL}/{fig_pdf}")
    _upload(fig_png, f"{REMOTE_FINAL}/{fig_png}")
    print(f"[upload] {REMOTE_FINAL}/{fig_pdf} + .png")

    # Delete cache folder on HF
    print("[cleanup] deleting cache on HF")
    for job in JOBS:
        try:
            api.delete_file(path_in_repo=f"{REMOTE_CACHE}/{job['result_name']}",
                            repo_id=HF_REPO, token=HF_TOKEN)
        except Exception as e:
            print(f"  could not delete {job['result_name']}: {e}")
    print("[done]")

if __name__ == "__main__":
    main()
