# Publication-Grade Q-ECV Reliability Pipeline

This package replaces the original notebook with a standalone, reproducible pipeline for the revised manuscript on hybrid quantum–classical chest X-ray classification under NISQ noise.

## Files

- `qecv_publication_pipeline.py` — complete training, evaluation, statistics, diagnostics, plots, and manuscript tables.
- `qecv_publication_config.json` — exhaustive five-seed publication configuration.
- `qecv_reproduce_original_config.json` — one-seed, five-epoch configuration matching the original training setup while correcting purity and statistics.
- `qecv_qiskit_backend_extension.py` — optional finite-shot evaluation with a generic Qiskit Aer noise model or a live IBM backend-derived calibration.
- `requirements_qecv.txt` — Python dependencies.
- `QECV_Publication_Pipeline_Colab.ipynb` — self-contained Google Colab version.

## Corrections implemented

The code explicitly separates the following quantities:

1. **True global quantum-state purity**

   \[
   \mathcal{P}_{\mathrm{global}}=\operatorname{Tr}(\rho^2).
   \]

   This is calculated analytically with `qml.purity` on `default.mixed`.

2. **Mean local-qubit purity**

   \[
   \mathcal{P}_{\mathrm{local}}
   =\frac{1}{n}\sum_i\frac{1+\langle X_i\rangle^2+\langle Y_i\rangle^2+\langle Z_i\rangle^2}{2}.
   \]

3. **Pauli-Z polarization score**

   \[
   S_Z=\frac{1}{n}\sum_i\langle Z_i\rangle^2.
   \]

   This is the quantity called “purity” in the earlier notebook. It is now correctly reported as a measurement-axis proxy and is never interpreted as `Tr(rho²)`.

The VQC-head ablation is fully trained. The pipeline also distinguishes clean training, amplitude-damping training, depolarizing-noise training, and randomly mixed-channel training.

## Dataset structure

The default mode expects:

```text
ASCE_final_processed/
├── bacteria/
├── normal/
└── virus/
```

Supported image types are those readable by Pillow and `torchvision.datasets.ImageFolder`.

For defensible patient-level splitting, prepare a CSV:

```csv
path,label,patient_id
/path/image1.jpeg,bacteria,P001
/path/image2.jpeg,normal,P002
/path/image3.jpeg,virus,P003
```

Set:

```json
"manifest_csv": "/path/manifest.csv",
"patient_level_split": true
```

A patient-level manifest is preferable to filename-based patient-ID inference.

## Installation in Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
pip install -r requirements_qecv.txt
```

## Corrected reproduction of the original experiment

```bash
python qecv_publication_pipeline.py \
  --config qecv_reproduce_original_config.json
```

This uses seed 42, five epochs, amplitude-damping training at γ = 0.02, the original noise grid, and the original saved split indices. The resulting global-purity values will differ from the earlier JSON because the old notebook calculated the Z-polarization proxy rather than `Tr(rho²)`.

## First validation run

Run a one-seed, one-epoch smoke test before launching the full experiment:

```bash
python qecv_publication_pipeline.py \
  --config qecv_publication_config.json \
  --quick-test
```

The quick test verifies:

- dataset discovery;
- saved split loading;
- model forward and backward passes;
- true-purity calculation;
- noise-sweep evaluation;
- calibration and prediction export;
- bootstrap and plot generation.

## Complete publication run

```bash
python qecv_publication_pipeline.py \
  --config qecv_publication_config.json
```

The supplied exhaustive configuration performs 75 training experiments:

- three classical models for five seeds;
- three quantum/hybrid models;
- four training-noise conditions;
- five seeds.

Each quantum experiment is evaluated under three noise channels, nine strengths, and analytic/1,024-shot/4,096-shot settings. This is computationally expensive. Preserve `"resume": true` so interrupted runs continue from checkpoints.

## Models

| Configuration name | Description |
|---|---|
| `classical_full` | DenseNet-121 1,024-dimensional representation with linear classifier |
| `classical_bottleneck` | DenseNet-121 → six-dimensional bounded bottleneck → classifier |
| `classical_mlp` | Parameter-matched six-dimensional classical nonlinear head |
| `vqc_head` | Trained VQC-head model without the classical skip branch |
| `hybrid_fixed` | Fixed concatenation of six classical angular and six quantum features |
| `hybrid_adaptive` | Reliability-guided quantum contribution using true purity and Z-polarization |

## Training conditions

| Condition | Training noise |
|---|---|
| `clean` | No quantum noise |
| `amplitude_002` | Amplitude damping, γ = 0.02 |
| `depolarizing_002` | Depolarizing noise, γ = 0.02 |
| `mixed_005` | Channel and strength sampled per batch up to γ = 0.05 |

Classical models are trained only in the clean condition because quantum-channel labels do not apply to them.

## Statistical analysis

For each condition, the code produces:

- accuracy, balanced accuracy, macro-F1, weighted-F1 and MCC;
- macro one-vs-rest AUROC and macro-AUPRC;
- ECE, Brier score and negative log-likelihood;
- precision, sensitivity, specificity, F1, AUROC and AUPRC by class;
- confusion matrices, ROC curves and precision–recall curves;
- 1,000-resample bootstrap confidence intervals;
- paired bootstrap differences between noisy and clean predictions;
- Holm-adjusted p-values;
- mean and t-based 95% confidence intervals across five training seeds;
- Pearson and Spearman diagnostic–performance associations;
- quadratic trend fit for non-monotonic responses;
- normalized robustness area under the noise-performance curve;
- risk–coverage analysis and area under the risk–coverage curve.

All paired tests use the same test cases and verify that prediction order is unchanged.

## Temperature scaling

A single temperature is estimated from clean validation logits and then fixed for every test noise condition. Both the learned temperature and calibrated per-sample probabilities are saved. This avoids fitting calibration parameters on the test set.

## Principal output files

```text
OUTPUT_DIR/
├── resolved_config.json
├── environment.json
├── split_summary.json
├── dataset_manifest_with_splits.csv
├── exact_duplicate_images.csv
├── cross_split_duplicates.json
├── all_experiment_metrics.csv
├── experiments/
│   └── MODEL__TRAINING_CONDITION__SEED/
│       ├── best_model.pt
│       ├── checkpoint_latest.pt
│       ├── history.csv
│       ├── evaluation_metrics.csv
│       ├── temperature.json
│       ├── predictions/*.csv
│       ├── statistics/
│       │   ├── bootstrap_confidence_intervals.csv
│       │   ├── paired_bootstrap_differences.csv
│       │   ├── purity_performance_correlations.csv
│       │   └── robustness_area_under_curve.csv
│       ├── selective_prediction/*.csv
│       └── figures/*.png
└── summary/
    ├── multi_seed_summary.csv
    ├── table_clean_performance.csv
    ├── table_noise_robustness.csv
    ├── table_finite_shot_results.csv
    └── cross_channel_auroc_at_gamma_005.csv
```

## Qiskit backend-derived evaluation

### Generic Aer noise model

```bash
python qecv_qiskit_backend_extension.py \
  --config qecv_publication_config.json \
  --checkpoint /path/to/best_model.pt \
  --model hybrid_fixed \
  --generic-noise \
  --shots 4096 \
  --output-dir /path/to/qiskit_generic_results
```

### Live IBM backend calibration

Store the API token in the environment rather than the source code:

```bash
export QISKIT_IBM_TOKEN="YOUR_TOKEN"
```

```bash
python qecv_qiskit_backend_extension.py \
  --config qecv_publication_config.json \
  --checkpoint /path/to/best_model.pt \
  --model hybrid_fixed \
  --backend-name YOUR_AVAILABLE_BACKEND \
  --shots 4096 \
  --output-dir /path/to/ibm_derived_results
```

The backend extension currently supports `hybrid_fixed` and `vqc_head`. Global density-matrix purity is not directly measurable from ordinary finite-shot hardware counts; the extension therefore reports the hardware-accessible Pauli-Z response and predictive metrics. Simulator-only global purity must remain clearly identified as such in the manuscript.

## Recommended execution sequence

1. Run `--quick-test`.
2. Run the three controlled classical baselines for all five seeds.
3. Train `vqc_head` and `hybrid_fixed` under the clean condition.
4. Add channel-specific and mixed-noise training.
5. Run `hybrid_adaptive` after the fixed-fusion baseline is stable.
6. Execute the analytic noise sweep.
7. Execute finite-shot evaluation.
8. Run the Qiskit backend-derived extension on the selected best model.
9. Base every manuscript table on generated CSV files rather than manually transcribed numbers.

## Reproducibility note

The original split arrays can be reused by setting `existing_split_dir`. The script verifies that they contain no overlapping indices and cover all 5,016 images exactly once. Patient-level leakage cannot be excluded from image indices alone; use a patient manifest when available.
