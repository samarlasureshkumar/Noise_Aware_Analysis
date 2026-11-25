# QECV: reproducibility results (staged release)

This package contains available sample-level prediction ledgers, model-seed metrics, statistical exports, and provenance records for six model families across seeds 11, 22 and 33.

- `results/primary/`: primary model-seed experiment artifacts.
- `results/summary/`: independently recomputed clean metrics.
- `provenance/frozen_cohort/`: authoritative image-level manifest and split records.
- `provenance/historical_cohort_audit/`: original failed cohort-gate evidence, **historical only**.
- `provenance/correction_audit/`: correction-related audit records.

**Interpretation boundaries:** The 18 clean model-seed predictions have been reported as cohort-aligned in the working audit. Noise-channel coverage, confidence intervals, and VQC seed-22 instability require separate scientific interpretation. The dataset uses image-level, not patient-level, splitting. Simulated channels do not establish quantum hardware performance or quantum advantage.

This release intentionally excludes raw X-rays, model checkpoints, and legacy alternate-cohort prediction files. Some paths in prediction ledgers may reveal original directory layouts; review before making this package public.
