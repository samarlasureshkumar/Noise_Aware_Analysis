QECV Enhanced Reliability Study
This package replaces the earlier QECV notebook with a reproducible pipeline for a revised manuscript on noise-aware reliability of hybrid quantum-classical chest X-ray classifiers.
What the code corrects
The earlier notebook calculated `mean(Z_expectation**2)` and labelled it quantum-state purity. This package instead obtains the full density matrix from PennyLane and verifies true purity as
[
\mathrm{Purity}=\operatorname{Tr}(\rho^2).
]
Every state-diagnostic run aborts unless:
zero-noise full-state purity is approximately 1;
`qml.purity` agrees with purity calculated from the density matrix;
purity remains within the physical interval ([1/2^n,1]);
phase damping preserves computational-basis populations while reducing coherence in the installation sanity test.
The code also properly trains the quantum-head model, separates the six-dimensional bottleneck effect from the quantum effect, adds noise-aware training, repeated seeds, structural ablations, calibration, paired bootstrap tests, and mechanism-oriented state analysis.
Package contents
`qecv_enhanced_pipeline.py`: complete implementation and command-line interface.
`QECV_Enhanced_Colab.ipynb`: staged Google Colab workflow.
`config_pilot.json`: small validation run before committing major compute.
`config_publication.json`: repeated-seed publication experiment plan.
`requirements_qecv.txt`: required Python packages other than the Colab-provided PyTorch stack.
`smoke_test_qecv.py`: installation, gradient, metric, and quantum-physics test.
Expected dataset structure
The default configuration uses the path from the earlier notebook:
```text
/content/drive/MyDrive/Lung_data/ASCE_final_processed/
├── bacteria/
├── normal/
└── virus/
```
The class folder names must match the order in `expected_classes`.
Patient-independent splitting
For a defensible medical-imaging study, provide a metadata CSV when patient identifiers are available:
```csv
relative_path,patient_id
bacteria/image_001.jpeg,P001
bacteria/image_002.jpeg,P001
normal/image_101.jpeg,P057
```
Set `metadata_csv` in the configuration. The pipeline then creates patient-group-disjoint train, validation, and test sets. Without metadata, it uses stratified image-level splitting and explicitly records that limitation.
Duplicate leakage audit
The preparation stage computes SHA-256 hashes and stops if identical image files occur across splits. If the dataset changes, delete the previous run's `splits/` directory before recreating the experiment.
Colab setup
Copy the complete package folder to:
```text
/content/drive/MyDrive/QECV_Enhanced/
```
Open `QECV_Enhanced_Colab.ipynb` in Colab.
Select the pilot profile first.
Confirm the dataset and output paths.
Run the smoke test and preparation stage.
Run each experimental stage separately so checkpoints can be resumed.
The pipeline caches 1,024-dimensional DenseNet features once. This avoids repeatedly running the image backbone during the large quantum experiment matrix.
Command-line execution
From the project directory:
```bash
pip install -r requirements_qecv.txt
python smoke_test_qecv.py
```
Pilot preparation:
```bash
python qecv_enhanced_pipeline.py \
  --config config_pilot.json \
  --stage prepare
```
Publication preparation:
```bash
python qecv_enhanced_pipeline.py \
  --config config_publication.json \
  --stage prepare
```
Run the remaining publication stages individually:
```bash
python qecv_enhanced_pipeline.py --config config_publication.json --stage controls
python qecv_enhanced_pipeline.py --config config_publication.json --stage training-noise
python qecv_enhanced_pipeline.py --config config_publication.json --stage structural
python qecv_enhanced_pipeline.py --config config_publication.json --stage diagnostics
python qecv_enhanced_pipeline.py --config config_publication.json --stage summarize
```
Use `--force` only when a completed checkpoint must be retrained. Without it, existing checkpoints are loaded and evaluation continues.
Experimental stages and reviewer concerns
Stage	Scientific purpose	Main reviewer issues addressed

`prepare`	Saved split indices, patient grouping, exact-duplicate audit, shared DenseNet training, feature cache, physical sanity checks	Reproducibility and clinical data integrity
`controls`	Full classical model, six-dimensional classical bottleneck, capacity-matched classical MLP, trained quantum head, and hybrid fusion	Quantum contribution, bottleneck confounding, invalid quantum-only ablation
`training-noise`	Clean, amplitude, phase, depolarizing, and mixed-noise training evaluated against every test-time channel	Training/inference mismatch and NISQ deployment robustness
`structural`	One-factor variations of qubits, layers, encoding, rotation block, and entanglement	Missing circuit-depth, qubit, encoding, and topology ablations
`diagnostics`	True purity, entropy, coherence, fidelity, trace distance, state-distribution drift, prediction drift, and risk-coverage	Shallow mechanism analysis and unexplained phase damping
`summarize`	Seed aggregation, confidence intervals, reliability diagrams, state–prediction scatter plots, risk–coverage curves, structural-ablation figures, training-noise matrix, and combined tables	Statistical evidence and presentation
Models included in the controls
`classical_full`: 1,024-dimensional DenseNet feature vector directly classified.
`classical_bottleneck`: 1,024 to (n) projection followed by a classical classifier.
`classical_matched_mlp`: bottleneck followed by a classical head approximately capacity-matched to the hybrid head.
`quantum_head`: DenseNet projection, VQC, and classifier without a direct classical bypass.
`hybrid`: projected classical features concatenated with VQC outputs.
For the hybrid checkpoint, the code additionally evaluates:
`q = 0`: quantum branch removed at inference;
shuffled `q`: quantum features detached from their corresponding samples;
classifier weight norms for classical and quantum fusion branches.
These experiments determine whether apparent robustness is genuine quantum contribution or merely reliance on the classical bypass.
Training-noise regimes
The publication configuration trains the main hybrid architecture under:
clean training;
fixed amplitude damping;
fixed phase damping;
fixed depolarizing noise;
mixed channel and random intensity training.
Each model is evaluated under all three channels and the full noise-strength grid. The same clean-validation temperature is retained during noisy testing to measure calibration drift rather than recalibrating each condition.
Structural ablation design
The pipeline uses one-factor-at-a-time variations around the main configuration:
qubits: 4, 6, and 8;
layers: 1, 2, and 4;
encoding: angle and data re-uploading;
entanglement: none, linear, and ring;
rotations: `RY` and `Rot`/XYZ.
This is substantially more computationally manageable and interpretable than a full Cartesian grid.
True state diagnostics
For a fixed stratified diagnostic subset, the code records per image and per noise condition:
true full-state purity;
von Neumann entropy;
normalized (l_1)-coherence;
fidelity to the corresponding ideal state;
trace distance from the ideal state;
Jensen-Shannon divergence of computational-basis populations;
correctly named squared local-(Z) magnetization proxy;
prediction-probability Jensen-Shannon divergence;
probability displacement and sample-level NLL change.
The zero-noise state is generated by the same learned circuit without a noise channel. Because this reference circuit is unitary, its full-state purity must be one.
Statistical outputs
The code produces:
stratified bootstrap 95% confidence intervals for all primary metrics;
paired bootstrap differences between clean and noisy predictions on the same test images;
primary-scope bootstrap execution by default (controls and the clean-trained main hybrid sweep), avoiding redundant resampling of every exploratory ablation;
two-sided bootstrap sign probabilities;
Holm-adjusted probabilities within each experiment and metric;
repeated-seed mean, standard deviation, and Student-(t) confidence intervals;
exploratory sample-level Spearman analyses between state degradation and prediction changes.
The small noise-associated AUROC increase should be described as a regularization effect only when its confidence interval and repeated-seed evidence support that interpretation.
Output structure
```text
qecv_enhanced_runs/<run_name>/
├── audit/
│   ├── exact_duplicate_audit.csv
│   └── quantum_sanity_checks.json
├── backbone/
├── checkpoints/
├── diagnostics/
├── features/
├── figures/
├── logs/
├── predictions/
├── splits/
├── tables/
├── study_config.json
└── summary_artifacts.json
```
Important tables include:
`evaluation_sweeps.csv`
`evaluation_seed_aggregate.csv`
`metric_bootstrap_ci.csv`
`paired_noise_bootstrap.csv`
`per_class_metrics.csv`
`confusion_matrices.csv`
`calibration_bins.csv`
`checkpoint_registry.csv`
`hybrid_branch_weight_analysis.csv`
`state_diagnostics_combined.csv`
`state_diagnostics_per_sample_combined.csv`
`structural_ablation_summary.csv`
`diagnostic_risk_coverage.csv`
Minimum publication run
Do not prepare a revised manuscript from the pilot output. A minimum confirmatory run should include:
all five publication seeds for the control and training-noise experiments;
all three test-time channels;
the complete strength grid;
three structural-ablation seeds;
true state diagnostics for at least 128 stratified test images, preferably the complete test set when computationally feasible;
patient-independent splitting when identifiers are available;
exact-duplicate leakage resolution;
bootstrap confidence intervals and paired comparisons.
Important interpretation rules
Do not call the local-(Z) proxy purity.
Do not claim quantum advantage when the classical baseline is superior.
Do not interpret unchanged (Z)-expectations under phase damping as unchanged full-state purity.
Do not call an untrained quantum head an ablation.
Do not infer a mechanism from a correlation across only a small set of noise strengths.
Describe canonical channel simulation accurately; it is not equivalent to executing on a specific quantum processor.
