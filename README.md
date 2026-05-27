# QPriv-VL — Question-Guided Token Pruning for Privacy-Preserving VQA

A research pipeline that applies privacy-aware visual token pruning to
LLaVA-1.5-7B across four distributed learning paradigms, then evaluates
the resulting privacy/utility trade-off under four adversarial attacks.

---

## Overview

The pipeline has three phases:

| Phase | Modules | Description |
|-------|---------|-------------|
| **Teacher training** | 01 – 04 | Extract features, find utility-floor rho, train sensitivity head + threshold predictor, distil into a lightweight student. |
| **Apply & fine-tune** | 05 – 09 | Visualise the student, then fine-tune LLaVA with LoRA under direct, federated, split, and U-shaped learning. |
| **Attack evaluation** | 10 – 14 | Shared attack infrastructure, FSHA, FORA, MIA, and gradient-leakage attacks against the trained checkpoints. |

---

## Repository structure

```
qgtp_project/
├── qgtp_lib.py                  # Shared library (all apply + attack modules import this)
├── module_01_data_prep.py       # Feature extraction
├── module_02_rho_search.py      # Utility-floor rho* search
├── module_03_sensitivity.py     # Sensitivity head + teacher predictor
├── module_04_student.py         # Student distillation
├── module_05_visualization.py   # Publication figures
├── module_06_direct_apply.py    # Centralized LoRA fine-tuning
├── module_07_federated_learning.py  # FedAvg
├── module_08_split_learning.py  # Split learning (single cut)
├── module_09_u_shaped.py        # U-shaped split learning (two cuts)
├── module_10_attack_common.py   # Shared attack utilities
├── module_11_fsha_original.py   # FSHA (Feature-Stealing/Hijack Attack)
├── module_12_attack_fora.py     # FORA (Feature Oracle Reconstruction Attack)
├── module_13_MIA.py             # Membership Inference Attack
└── module_14_attack_grad_leak.py # iDLG gradient-leakage attack
```

---

## Requirements

```bash
pip install torch torchvision transformers datasets peft \
            huggingface_hub tqdm numpy matplotlib pillow \
            scikit-learn accelerate
```

A CUDA GPU with at least 40 GB VRAM is required for modules 02, 06–14.
Modules 01, 03, 04 require at least 24 GB.

---

## Configuration

All modules read credentials and paths from **environment variables**.
Set these before running any module:

```bash
export HF_USER="your-huggingface-username"
export HF_TOKEN="hf_..."           # read/write token for your HF repo
export HF_REPO="${HF_USER}/QprivVL"  # or any repo name you prefer
export RESULTS_DIR="./results"       # local output directory
```

For offline / cached runs set:

```bash
export OFFLINE=1
```

---

## Running the pipeline

### Phase 1 — Teacher training (run once)

**Step 1 — Data preparation** (one run per dataset)

```bash
DATASET_KEY=vqarad python module_01_data_prep.py
DATASET_KEY=slake  python module_01_data_prep.py
# repeat for: vqav2 gqa okvqa pathvqa
```

Extracts CLIP-ViT-L/14@336 and DINOv2-S/14 features per sample and uploads
`features_<dataset>.pt` to the HF Hub.

**Step 2 — Rho* search**

```bash
python module_02_rho_search.py
```

For each dataset, finds the smallest pruning ratio rho at which LLaVA still
produces the correct answer.  Uploads `rho_star_<dataset>.pt`.

**Step 3 — Sensitivity head + teacher predictor**

```bash
python module_03_sensitivity.py
```

Trains a per-patch privacy scorer (DINOv2 → sensitivity) and a threshold
predictor that maps (CLIP features, sensitivity summary) → rho.
Uploads `sensitivity_head.pt` and `teacher_predictor.pt`.

**Step 4 — Student distillation**

```bash
python module_04_student.py
```

Distils the teacher into a lightweight student (~5 M params).
Uploads `student_model.pt`.

---

### Phase 2 — Visualisation and fine-tuning

**Step 5 — Publication figures**

```bash
python module_05_visualization.py
```

Generates five figure types (pipeline decomposition, teacher vs. student,
rho agreement scatter, kept-token distribution, per-dataset scatter) and
uploads them under `viz_paper/` on the HF Hub.

**Steps 6–9 — Fine-tuning**

Each module accepts `DATASET_KEY`, `QGTP_MODE` (`off` / `fixed` / `student`),
and `FIXED_RHO` as environment variables.

```bash
# Direct (centralized)
DATASET_KEY=vqarad QGTP_MODE=off     python module_06_direct_apply.py
DATASET_KEY=vqarad QGTP_MODE=fixed FIXED_RHO=0.5  python module_06_direct_apply.py
DATASET_KEY=vqarad QGTP_MODE=student python module_06_direct_apply.py

# Federated (FedAvg, 5 clients)
DATASET_KEY=vqarad QGTP_MODE=off python module_07_federated_learning.py

# Split learning (cut at LM layer 16)
DATASET_KEY=vqarad QGTP_MODE=off python module_08_split_learning.py

# U-shaped split learning (cuts at layers 8 and 24)
DATASET_KEY=vqarad QGTP_MODE=off python module_09_u_shaped.py
```

Each run saves:
- `results/lora_<tag>.pt` — best LoRA weights
- `results/results_<tag>.json` — full training history
- `results/fig_loss_<tag>.png`, `fig_acc_<tag>.png` — training curves
- Uploads all three under `lora/`, `results/`, and `figures/` on the HF Hub

---

### Phase 3 — Attack evaluation

All attack modules share `module_10_attack_common.py` for data loading,
smashed-feature extraction, attack model architectures, and utilities.

**Module 11 — FSHA (Feature-Stealing/Hijack Attack)**

Simulates the end-state of FSHA on split-learning checkpoints: trains a
shadow autoencoder adversarially aligned to real smashed activations via a
discriminator, then inverts private activations into images.

```bash
python module_11_fsha_original.py
```

Outputs: `results_fsha_orig_split_vqarad.json`, qualitative figure, and
28 individual reconstruction images.

**Module 12 — FORA (Feature Oracle Reconstruction Attack)**

Trains one oracle decoder on unpruned (off) public smashed activations, then
evaluates it against all defense settings. Establishes the reconstruction
upper bound for the attacker.

```bash
MODEL_TAG=split_vqarad_off_cut16 python module_12_attack_fora.py
# Other tags: direct_vqarad_off, federated_vqarad_student, ushape_slake_off_a8b24 …
```

Outputs: PSNR / cosine curves vs. rho, qualitative figure.

**Module 13 — MIA (Membership Inference Attack)**

Trains a small transformer classifier on smashed activations to distinguish
open-ended from closed-ended (yes/no) questions — a proxy attribute that
reveals private data characteristics.

```bash
python module_13_MIA.py
```

Evaluates all 24 (paradigm × defense) combinations with caching/resume.
Outputs: ROC-AUC and accuracy curves per paradigm.

**Module 14 — Gradient leakage (iDLG)**

Optimises a dummy visual feature to match real LoRA gradients via
normalized per-tensor L2 matching.

```bash
python module_14_attack_grad_leak.py
```

Reports PSNR and cosine similarity on the full 576×1024 feature tensor
(primary) and on the defender's kept rows (secondary).
Outputs: bar chart and convergence diagnostic figure.

---

## Key design choices

| Choice | Rationale |
|--------|-----------|
| CLIP-ViT-L/14@336 for both vision and text | Aligned joint embedding space for the QGTP cross-attention scorer. |
| DINOv2-S/14 for sensitivity | Generalises across medical, natural, and remote-sensing domains without domain-specific supervision. |
| Normalized L2 gradient matching (module 14) | Scale-invariant; avoids the cap-at-2 artefact of cosine similarity. |
| Primary metric on full 576×1024 feature (module 14) | Pruned rows that stay at noise are a defence win, not a measurement gap. |

---

## Outputs summary

| Artefact | Location (HF Hub) |
|----------|-------------------|
| Features | `features_<dataset>.pt` |
| Rho search | `rho_star_<dataset>.pt` |
| Sensitivity head | `sensitivity_head.pt` |
| Teacher predictor | `teacher_predictor.pt` |
| Student model | `student_model.pt` |
| LoRA checkpoints | `lora/lora_<tag>.pt` |
| Training results | `results_<tag>.json` |
| Figures | `figures/`, `viz_paper/` |
| Attack results | `FSHA_Updated/`, `GRADLEAK_Updated/`, `updated_MIA/` |
