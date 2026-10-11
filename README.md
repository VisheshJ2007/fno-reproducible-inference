# Reproducible FNO-2D Inference on Georgia Tech PACE ICE Cluster

This repository contains the complete reproducible inference and validation pipeline for a 2D Fourier Neural Operator (FNO-2D) surrogate model applied to Darcy Flow steady-state pressure field benchmarks on the Georgia Tech AI Makerspace / PACE ICE cluster

---

## 1. System & Hardware Provenance
- **Compute Cluster:** Georgia Tech PACE ICE (login-ice.pace.gatech.edu)
- **Compute Node:** atl1-1-01-005-9-0.pace.gatech.edu
- **GPU Hardware:** NVIDIA A40 (46,068 MiB VRAM, Driver 615.71.09, CUDA 13.0)
- **Slurm Job ID:** 6157714 / 6157931 (Partition: coc-gpu, QOS: coc-ice, Account: ece)

---

## 2. Model & Data Artifacts
- **Model Checkpoint:** models/fno_darcy_checkpoint.pt
  * Architecture: FNO-2D (n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1)
  * Parameters: Truncated low-frequency Fourier spectral modes capturing fluid pressure gradients.
- **Evaluation Dataset:** data/darcy_test_sample.pt
  * 16 evaluation instances at 16x16 resolution extracted from Darcy Flow small benchmarks.
  * Inputs (x): Permeability field tensor [16, 1, 16, 16].
  * Ground Truth (y): Classical numerical solver pressure field tensor [16, 1, 16, 16].

---

## 3. Environment Setup & Reproduction

To reproduce this environment from source:

```bash
# Clone the repository
git clone https://github.com/VisheshJ2007/fno-reproducible-inference.git
cd fno-reproducible-inference

# Recreate Conda environment
conda env create -f environment.yml
conda activate fno-inference
```

---

## 4. Execution Workflow

### GPU Batch Submission via SLURM
```bash
sbatch scripts/submit_inference.sbatch
```

### Standalone Inference Verification
```bash
python scripts/run_inference.py
```

---

## 5. Validation Gates & Results

The pipeline enforces two verification gates documented in outputs/smoke_test_output.json:
,
1. Smoke Test (Single Forward Pass):
  * Output Tensor Dimension: [1, 1, 16, 16] (Matches ground truth)
  * Single-sample Latency: 190.39 ms
  * Status: PASSED

2. Substantive Scientific Validation:
  * Metric: Relative L2 Error Norm (||u_pred - u_true||_2 / ||u_true||_2)
  * Mean Relative L2 Error: 0.07770 (Std: 0.01828)
  * Mean Squared Error (MSE): 0.001452
  * Threshold Criterion: Relative L2 < 0.20
  * Status: PASSED

3. Hardware Acceleration Comparison:
  * CPU Batch Latency: 3.50 s
  * NVIDIA A40 GPU Batch Latency: 0.0682 s (>50x hardware acceleration)
