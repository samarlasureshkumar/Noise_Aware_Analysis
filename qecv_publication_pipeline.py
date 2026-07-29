#!/usr/bin/env python3
"""
Publication-grade hybrid quantum-classical reliability pipeline for three-class
chest X-ray classification.

This standalone script corrects the methodological problems found in the original
notebook and produces the analyses needed for a journal manuscript:

1. Reproducible image-level or patient-level train/validation/test splits.
2. Controlled classical and quantum-classical baselines.
3. Clean, channel-specific, and mixed-channel noise-aware training.
4. Physically valid full-state purity Tr(rho^2) from PennyLane default.mixed.
5. Mean local-qubit purity and the original Pauli-Z polarization proxy, reported
   as distinct quantities.
6. Analytic and finite-shot inference under amplitude damping, phase damping,
   and depolarizing noise.
7. Temperature scaling, calibration metrics, reliability diagrams, and ECE.
8. Per-class metrics, ROC/PR curves, confusion matrices, and prediction exports.
9. Multi-seed aggregation, non-parametric bootstrap confidence intervals,
   paired bootstrap differences, Holm correction, and robustness AUC.
10. Risk-coverage/selective prediction analyses using confidence, entropy,
    global purity, local purity, and Z-polarization.
11. Purity-guided adaptive fusion as an optional method-level contribution.

Expected ImageFolder layout
---------------------------
DATA_ROOT/
    bacteria/*.png (or jpg/jpeg)
    normal/*.png
    virus/*.png

Optional CSV manifest columns
-----------------------------
path,label,patient_id

The script intentionally does not claim quantum advantage. It is designed to
measure robustness and reliability under controlled simulated noise.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import scipy
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from PIL import Image
from scipy.stats import pearsonr, spearmanr, t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_default_dtype(torch.float32)

SUPPORTED_NOISE = ("none", "amplitude", "phase", "depolarizing")
QUANTUM_MODEL_NAMES = {"vqc_head", "hybrid_fixed", "hybrid_adaptive"}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainCondition:
    name: str
    noise_type: str = "none"
    noise_strength: float = 0.0
    mixed_noise_types: List[str] = field(
        default_factory=lambda: ["amplitude", "phase", "depolarizing"]
    )
    mixed_max_strength: float = 0.05


@dataclass
class Config:
    # Paths
    data_root: str = "/content/drive/MyDrive/Lung_data/ASCE_final_processed"
    manifest_csv: Optional[str] = None
    output_dir: str = "/content/drive/MyDrive/qecv_publication_runs"
    existing_split_dir: Optional[str] = None

    # Data
    class_names: List[str] = field(
        default_factory=lambda: ["bacteria", "normal", "virus"]
    )
    image_size: int = 224
    batch_size: int = 2
    num_workers: int = 2
    pin_memory: bool = True
    test_fraction: float = 0.15
    val_fraction_of_remaining: float = 0.15
    patient_level_split: bool = False
    duplicate_hash_check: bool = True
    use_horizontal_flip: bool = True
    rotation_degrees: float = 5.0
    color_jitter: float = 0.10

    # Model
    pretrained_backbone: bool = True
    n_qubits: int = 6
    q_layers: int = 2
    bottleneck_dim: int = 6
    freeze_backbone_epochs: int = 0
    adaptive_gate_inputs: List[str] = field(
        default_factory=lambda: ["global_purity", "z_polarization"]
    )
    quantum_device_seed: int = 42

    # Training
    seeds: List[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    early_stopping_patience: int = 4
    selection_metric: str = "auroc_ovr"
    class_weighted_loss: bool = True
    balanced_sampler: bool = False
    amp: bool = False

    # Models and training conditions
    model_variants: List[str] = field(
        default_factory=lambda: [
            "classical_full",
            "classical_bottleneck",
            "classical_mlp",
            "vqc_head",
            "hybrid_fixed",
            "hybrid_adaptive",
        ]
    )
    train_conditions: List[TrainCondition] = field(
        default_factory=lambda: [
            TrainCondition(name="clean", noise_type="none", noise_strength=0.0),
            TrainCondition(
                name="amplitude_002", noise_type="amplitude", noise_strength=0.02
            ),
            TrainCondition(
                name="depolarizing_002",
                noise_type="depolarizing",
                noise_strength=0.02,
            ),
            TrainCondition(
                name="mixed_005",
                noise_type="mixed",
                noise_strength=0.0,
                mixed_max_strength=0.05,
            ),
        ]
    )

    # Evaluation
    eval_noise_types: List[str] = field(
        default_factory=lambda: ["amplitude", "phase", "depolarizing"]
    )
    noise_strengths: List[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    )
    shot_settings: List[Optional[int]] = field(default_factory=lambda: [None, 1024, 4096])
    compute_quantum_diagnostics: bool = True
    diagnostics_max_samples: Optional[int] = None

    # Calibration
    temperature_scaling: bool = True
    temperature_steps: int = 250
    temperature_lr: float = 0.05
    ece_bins: int = 15

    # Statistics
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 2026
    bootstrap_metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "macro_f1",
            "auroc_ovr",
            "auprc_macro",
            "ece",
            "brier",
            "nll",
        ]
    )
    alpha: float = 0.05

    # Selective prediction
    risk_coverage_points: int = 50
    minimum_selective_samples: int = 20

    # Runtime controls
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: bool = True
    overwrite: bool = False
    quick_test: bool = False

    def validate(self) -> None:
        if self.n_qubits != self.bottleneck_dim:
            raise ValueError("n_qubits must equal bottleneck_dim for angle encoding.")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        if not 0 < self.val_fraction_of_remaining < 1:
            raise ValueError("val_fraction_of_remaining must be between 0 and 1.")
        unknown_models = set(self.model_variants) - {
            "classical_full",
            "classical_bottleneck",
            "classical_mlp",
            "vqc_head",
            "hybrid_fixed",
            "hybrid_adaptive",
        }
        if unknown_models:
            raise ValueError(f"Unknown model variants: {sorted(unknown_models)}")
        for noise in self.eval_noise_types:
            if noise not in SUPPORTED_NOISE:
                raise ValueError(f"Unsupported evaluation noise type: {noise}")
        for condition in self.train_conditions:
            if condition.noise_type not in (*SUPPORTED_NOISE, "mixed"):
                raise ValueError(f"Unsupported training noise type: {condition.noise_type}")
        if self.quick_test:
            self.seeds = self.seeds[:1]
            self.epochs = min(self.epochs, 1)
            self.noise_strengths = [0.0, 0.05]
            self.shot_settings = [None]
            self.bootstrap_resamples = min(self.bootstrap_resamples, 100)
            self.model_variants = ["classical_bottleneck", "hybrid_fixed"]
            self.train_conditions = [
                TrainCondition(name="clean", noise_type="none", noise_strength=0.0)
            ]


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    raw_conditions = raw.pop("train_conditions", None)
    cfg = Config(**raw)
    if raw_conditions is not None:
        cfg.train_conditions = [TrainCondition(**item) for item in raw_conditions]
    cfg.validate()
    return cfg


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


# =============================================================================
# Reproducibility and environment
# =============================================================================

def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def worker_seed_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_environment() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pennylane": qml.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# =============================================================================
# Dataset and split management
# =============================================================================

class ManifestDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        class_to_idx: Mapping[str, int],
        transform: Optional[Any] = None,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.class_to_idx = dict(class_to_idx)
        self.classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        self.transform = transform
        required = {"path", "label"}
        missing = required - set(self.records.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, str]:
        row = self.records.iloc[index]
        path = str(row["path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        label = row["label"]
        label_index = int(label) if str(label).isdigit() else self.class_to_idx[str(label)]
        return image, label_index, index, path


class ImageFolderWithMetadata(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, str]:
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, int(label), index, path


def make_transforms(cfg: Config) -> Tuple[Any, Any]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_ops: List[Any] = [transforms.Resize((cfg.image_size, cfg.image_size))]
    if cfg.use_horizontal_flip:
        train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.rotation_degrees > 0:
        train_ops.append(transforms.RandomRotation(cfg.rotation_degrees))
    if cfg.color_jitter > 0:
        train_ops.append(
            transforms.ColorJitter(
                brightness=cfg.color_jitter, contrast=cfg.color_jitter
            )
        )
    train_ops.extend([transforms.ToTensor(), normalize])
    eval_ops = [
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def load_manifest_or_imagefolder(
    cfg: Config, transform: Any
) -> Tuple[Dataset, np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    class_to_idx = {name: idx for idx, name in enumerate(cfg.class_names)}
    if cfg.manifest_csv:
        frame = pd.read_csv(cfg.manifest_csv)
        if not frame["path"].map(os.path.isabs).all():
            base = Path(cfg.manifest_csv).resolve().parent
            frame["path"] = frame["path"].map(lambda p: str((base / str(p)).resolve()))
        dataset = ManifestDataset(frame, class_to_idx, transform=transform)
        labels = frame["label"].map(
            lambda value: int(value)
            if str(value).isdigit()
            else class_to_idx[str(value)]
        ).to_numpy(dtype=int)
        groups = (
            frame["patient_id"].astype(str).to_numpy()
            if "patient_id" in frame.columns
            else np.array([f"image_{i}" for i in range(len(frame))])
        )
        paths = frame["path"].astype(str).tolist()
        return dataset, labels, groups, paths, class_to_idx

    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    dataset = ImageFolderWithMetadata(str(root), transform=transform)
    actual = dataset.class_to_idx
    if set(actual) != set(cfg.class_names):
        raise ValueError(
            f"Dataset folders {sorted(actual)} do not match configured classes "
            f"{sorted(cfg.class_names)}"
        )
    labels = np.array([label for _, label in dataset.samples], dtype=int)
    paths = [path for path, _ in dataset.samples]
    groups = np.array([infer_patient_id(path) for path in paths])
    return dataset, labels, groups, paths, actual


def infer_patient_id(path: str) -> str:
    """Best-effort ID extraction; a real manifest is preferred for patient-level splitting."""
    stem = Path(path).stem
    patterns = [r"^(person\d+)", r"^(patient[_-]?\d+)", r"^([A-Za-z]+\d+)"]
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return stem


def class_distribution(labels: np.ndarray, indices: np.ndarray, names: Sequence[str]) -> Dict[str, int]:
    counts = np.bincount(labels[indices], minlength=len(names))
    return {name: int(counts[i]) for i, name in enumerate(names)}


def split_objective(
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_fraction: float,
) -> float:
    overall = np.bincount(labels, minlength=int(labels.max()) + 1) / len(labels)
    test_dist = np.bincount(labels[test_idx], minlength=len(overall)) / max(len(test_idx), 1)
    fraction_error = abs(len(test_idx) / len(labels) - target_fraction)
    distribution_error = np.abs(test_dist - overall).mean()
    return float(fraction_error + distribution_error)


def best_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
    seed: int,
    attempts: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    best: Optional[Tuple[np.ndarray, np.ndarray]] = None
    best_score = float("inf")
    for offset in range(attempts):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=seed + offset
        )
        train_idx, test_idx = next(splitter.split(indices, labels, groups))
        if len(np.unique(labels[train_idx])) < len(np.unique(labels)):
            continue
        if len(np.unique(labels[test_idx])) < len(np.unique(labels)):
            continue
        score = split_objective(labels, train_idx, test_idx, test_fraction)
        if score < best_score:
            best_score = score
            best = (train_idx, test_idx)
    if best is None:
        raise RuntimeError("Unable to obtain a patient-level split containing every class.")
    return best


def create_or_load_splits(
    cfg: Config,
    labels: np.ndarray,
    groups: np.ndarray,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = Path(cfg.existing_split_dir) if cfg.existing_split_dir else output_dir / "splits"
    train_path = source / "train_idx.npy"
    val_path = source / "val_idx.npy"
    test_path = source / "test_idx.npy"
    if train_path.exists() and val_path.exists() and test_path.exists():
        train_idx = np.load(train_path)
        val_idx = np.load(val_path)
        test_idx = np.load(test_path)
    else:
        all_idx = np.arange(len(labels))
        if cfg.patient_level_split:
            trainval_rel, test_idx = best_group_split(
                labels, groups, cfg.test_fraction, cfg.seeds[0]
            )
            train_rel, val_rel = best_group_split(
                labels[trainval_rel],
                groups[trainval_rel],
                cfg.val_fraction_of_remaining,
                cfg.seeds[0] + 10_000,
            )
            train_idx = trainval_rel[train_rel]
            val_idx = trainval_rel[val_rel]
        else:
            first = StratifiedShuffleSplit(
                n_splits=1, test_size=cfg.test_fraction, random_state=cfg.seeds[0]
            )
            trainval_idx, test_idx = next(first.split(all_idx, labels))
            second = StratifiedShuffleSplit(
                n_splits=1,
                test_size=cfg.val_fraction_of_remaining,
                random_state=cfg.seeds[0] + 1,
            )
            train_rel, val_rel = next(
                second.split(np.arange(len(trainval_idx)), labels[trainval_idx])
            )
            train_idx = trainval_idx[train_rel]
            val_idx = trainval_idx[val_rel]

        split_dir = output_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "train_idx.npy", train_idx)
        np.save(split_dir / "val_idx.npy", val_idx)
        np.save(split_dir / "test_idx.npy", test_idx)

    validate_splits(train_idx, val_idx, test_idx, len(labels), groups, cfg.patient_level_split)
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)


def validate_splits(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_total: int,
    groups: np.ndarray,
    patient_level: bool,
) -> None:
    sets = [set(map(int, x)) for x in (train_idx, val_idx, test_idx)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Split indices overlap.")
    union = sets[0] | sets[1] | sets[2]
    if len(union) != n_total:
        raise ValueError(f"Splits cover {len(union)} of {n_total} samples.")
    if patient_level:
        group_sets = [set(groups[x]) for x in (train_idx, val_idx, test_idx)]
        if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
            raise ValueError("Patient/group leakage detected across splits.")


def sha256_file(path: str, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def dataset_audit(
    cfg: Config,
    paths: Sequence[str],
    labels: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    split_lookup = {}
    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        for index in indices:
            split_lookup[int(index)] = name
    hash_lookup: Dict[str, List[int]] = defaultdict(list)
    for index, path in enumerate(paths):
        row = {
            "sample_index": index,
            "path": path,
            "label_index": int(labels[index]),
            "label": cfg.class_names[int(labels[index])],
            "patient_id": str(groups[index]),
            "split": split_lookup[index],
        }
        if cfg.duplicate_hash_check:
            digest = sha256_file(path)
            row["sha256"] = digest
            hash_lookup[digest].append(index)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "dataset_manifest_with_splits.csv", index=False)
    if cfg.duplicate_hash_check:
        duplicates = [indices for indices in hash_lookup.values() if len(indices) > 1]
        duplicate_rows = []
        for group_id, indices in enumerate(duplicates):
            for index in indices:
                duplicate_rows.append(
                    {
                        "duplicate_group": group_id,
                        "sample_index": index,
                        "path": paths[index],
                        "split": split_lookup[index],
                    }
                )
        pd.DataFrame(duplicate_rows).to_csv(
            output_dir / "exact_duplicate_images.csv", index=False
        )
        cross_split = []
        for indices in duplicates:
            splits = {split_lookup[index] for index in indices}
            if len(splits) > 1:
                cross_split.append({"indices": indices, "splits": sorted(splits)})
        save_json(cross_split, output_dir / "cross_split_duplicates.json")


def build_loaders(
    cfg: Config,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, List[str], np.ndarray, Dict[str, int]]:
    train_transform, eval_transform = make_transforms(cfg)
    train_dataset, labels, groups, paths, class_to_idx = load_manifest_or_imagefolder(
        cfg, train_transform
    )
    eval_dataset, labels_eval, _, paths_eval, _ = load_manifest_or_imagefolder(
        cfg, eval_transform
    )
    if not np.array_equal(labels, labels_eval) or paths != paths_eval:
        raise RuntimeError("Training and evaluation datasets have different sample ordering.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = True
    if cfg.balanced_sampler:
        train_labels = labels[train_idx]
        counts = np.bincount(train_labels, minlength=len(cfg.class_names))
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_idx),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        worker_init_fn=worker_seed_fn,
        generator=generator,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_idx), shuffle=shuffle, sampler=sampler, **common
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_idx), shuffle=False, **common
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_idx), shuffle=False, **common
    )
    return train_loader, val_loader, test_loader, labels, paths, groups, class_to_idx


# =============================================================================
# Quantum circuit factory and diagnostics
# =============================================================================

def validate_noise(noise_type: str, strength: float) -> None:
    if noise_type not in SUPPORTED_NOISE:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("Noise strength must be in [0, 1].")


def apply_channel(noise_type: str, strength: float, wires: Iterable[int]) -> None:
    if noise_type == "none" or strength <= 0:
        return
    for wire in wires:
        if noise_type == "amplitude":
            qml.AmplitudeDamping(strength, wires=wire)
        elif noise_type == "phase":
            qml.PhaseDamping(strength, wires=wire)
        elif noise_type == "depolarizing":
            qml.DepolarizingChannel(strength, wires=wire)
        else:
            raise ValueError(noise_type)


def quantum_ansatz(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    n_qubits: int,
    q_layers: int,
    noise_type: str,
    noise_strength: float,
) -> None:
    for wire in range(n_qubits):
        qml.RY(inputs[wire], wires=wire)
    index = 0
    for _ in range(q_layers):
        for wire in range(n_qubits):
            qml.RY(weights[index], wires=wire)
            index += 1
        for wire in range(n_qubits):
            qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
        apply_channel(noise_type, noise_strength, range(n_qubits))


@lru_cache(maxsize=128)
def make_feature_qnode(
    n_qubits: int,
    q_layers: int,
    noise_type: str,
    shots: Optional[int],
    device_seed: int,
):
    validate_noise(noise_type, 0.0)
    device = qml.device("default.mixed", wires=n_qubits, shots=shots, seed=device_seed)
    diff_method = "backprop" if shots is None else None

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor, strength: float):
        quantum_ansatz(inputs, weights, n_qubits, q_layers, noise_type, strength)
        return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits))

    return circuit


@lru_cache(maxsize=64)
def make_purity_qnode(n_qubits: int, q_layers: int, noise_type: str, device_seed: int):
    """Analytic full-system purity. qml.purity is not a finite-shot observable."""
    device = qml.device("default.mixed", wires=n_qubits, shots=None, seed=device_seed)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor, strength: float):
        quantum_ansatz(inputs, weights, n_qubits, q_layers, noise_type, strength)
        return qml.purity(wires=tuple(range(n_qubits)))

    return circuit


@lru_cache(maxsize=192)
def make_axis_qnode(
    n_qubits: int,
    q_layers: int,
    noise_type: str,
    axis: str,
    shots: Optional[int],
    device_seed: int,
):
    device = qml.device("default.mixed", wires=n_qubits, shots=shots, seed=device_seed)
    observable = {"x": qml.PauliX, "y": qml.PauliY, "z": qml.PauliZ}[axis]
    diff_method = "backprop" if shots is None else None

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor, strength: float):
        quantum_ansatz(inputs, weights, n_qubits, q_layers, noise_type, strength)
        return tuple(qml.expval(observable(wire)) for wire in range(n_qubits))

    return circuit


def stack_qnode_output(values: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(values):
        tensor = values
    else:
        tensor = torch.stack(
            [value if torch.is_tensor(value) else torch.tensor(value) for value in values]
        )
    return tensor.to(device=device, dtype=torch.float32)


def z_polarization_score(q_out: torch.Tensor) -> torch.Tensor:
    """Original proxy, correctly named. This is not Tr(rho^2)."""
    return q_out.square().mean(dim=1)


def local_purity_from_bloch(
    x_expectation: torch.Tensor,
    y_expectation: torch.Tensor,
    z_expectation: torch.Tensor,
) -> torch.Tensor:
    per_qubit = 0.5 * (
        1.0
        + x_expectation.square()
        + y_expectation.square()
        + z_expectation.square()
    )
    return per_qubit.mean(dim=1)


# =============================================================================
# Models
# =============================================================================

class DenseNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        try:
            network = models.densenet121(weights=weights)
        except Exception as exc:
            warnings.warn(f"Pretrained DenseNet download failed ({exc}); using random weights.")
            network = models.densenet121(weights=None)
        self.features = network.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 1024

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = torch.relu(features)
        return self.pool(features).flatten(1)

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = trainable


class BaseClassifier(nn.Module):
    is_quantum: bool = False

    def set_backbone_trainable(self, trainable: bool) -> None:
        if hasattr(self, "backbone"):
            self.backbone.set_trainable(trainable)


class ClassicalFull(BaseClassifier):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.backbone = DenseNetBackbone(cfg.pretrained_backbone)
        self.classifier = nn.Linear(1024, len(cfg.class_names))

    def forward(self, images: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {"logits": self.classifier(features)}


class ClassicalBottleneck(BaseClassifier):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.backbone = DenseNetBackbone(cfg.pretrained_backbone)
        self.projection = nn.Linear(1024, cfg.bottleneck_dim)
        self.norm = nn.LayerNorm(cfg.bottleneck_dim)
        self.classifier = nn.Linear(cfg.bottleneck_dim, len(cfg.class_names))

    def forward(self, images: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        z_raw = self.projection(features)
        z_angle = math.pi * torch.tanh(z_raw)
        return {
            "logits": self.classifier(self.norm(z_angle)),
            "z_raw": z_raw,
            "z_angle": z_angle,
        }


class ClassicalMLP(BaseClassifier):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.backbone = DenseNetBackbone(cfg.pretrained_backbone)
        self.projection = nn.Linear(1024, cfg.bottleneck_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(cfg.bottleneck_dim),
            nn.Linear(cfg.bottleneck_dim, cfg.bottleneck_dim),
            nn.GELU(),
            nn.Linear(cfg.bottleneck_dim, len(cfg.class_names)),
        )

    def forward(self, images: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        z_raw = self.projection(features)
        z_angle = math.pi * torch.tanh(z_raw)
        return {"logits": self.mlp(z_angle), "z_raw": z_raw, "z_angle": z_angle}


class QuantumBase(BaseClassifier):
    is_quantum = True

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DenseNetBackbone(cfg.pretrained_backbone)
        self.projection = nn.Linear(1024, cfg.n_qubits)
        self.q_weights = nn.Parameter(
            torch.randn(cfg.q_layers * cfg.n_qubits, dtype=torch.float32) * 0.1
        )

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        z_raw = self.projection(features)
        z_angle = math.pi * torch.tanh(z_raw)
        return z_raw, z_angle

    def quantum_features(
        self,
        z_angle: torch.Tensor,
        noise_type: str,
        noise_strength: float,
        shots: Optional[int],
    ) -> torch.Tensor:
        validate_noise(noise_type, noise_strength)
        qnode = make_feature_qnode(
            self.cfg.n_qubits, self.cfg.q_layers, noise_type, shots, self.cfg.quantum_device_seed
        )
        outputs = [
            stack_qnode_output(qnode(row, self.q_weights, noise_strength), row.device)
            for row in z_angle
        ]
        return torch.stack(outputs, dim=0)

    def quantum_diagnostics(
        self,
        z_angle: torch.Tensor,
        q_out: torch.Tensor,
        noise_type: str,
        noise_strength: float,
        shots: Optional[int],
        compute_full: bool,
    ) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {
            "z_polarization": z_polarization_score(q_out)
        }
        if not compute_full:
            return diagnostics

        purity_node = make_purity_qnode(
            self.cfg.n_qubits, self.cfg.q_layers, noise_type, self.cfg.quantum_device_seed
        )
        global_purity = []
        for row in z_angle:
            value = purity_node(row, self.q_weights, noise_strength)
            global_purity.append(
                (value if torch.is_tensor(value) else torch.tensor(value)).to(
                    row.device, dtype=torch.float32
                )
            )
        diagnostics["global_purity"] = torch.stack(global_purity)

        axes: Dict[str, torch.Tensor] = {"z": q_out}
        for axis in ("x", "y"):
            node = make_axis_qnode(
                self.cfg.n_qubits,
                self.cfg.q_layers,
                noise_type,
                axis,
                shots,
                self.cfg.quantum_device_seed,
            )
            values = [
                stack_qnode_output(node(row, self.q_weights, noise_strength), row.device)
                for row in z_angle
            ]
            axes[axis] = torch.stack(values, dim=0)
        diagnostics["local_purity"] = local_purity_from_bloch(
            axes["x"], axes["y"], axes["z"]
        )
        return diagnostics


class VQCHead(QuantumBase):
    """DenseNet projection followed only by VQC features; no classical skip fusion."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.classifier = nn.Linear(cfg.n_qubits, len(cfg.class_names))

    def forward(
        self,
        images: torch.Tensor,
        noise_type: str = "none",
        noise_strength: float = 0.0,
        shots: Optional[int] = None,
        compute_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        z_raw, z_angle = self.encode(images)
        q_out = self.quantum_features(z_angle, noise_type, noise_strength, shots)
        diagnostics = self.quantum_diagnostics(
            z_angle,
            q_out,
            noise_type,
            noise_strength,
            shots,
            compute_diagnostics,
        )
        return {
            "logits": self.classifier(q_out),
            "z_raw": z_raw,
            "z_angle": z_angle,
            "q_out": q_out,
            **diagnostics,
        }


class HybridFixed(QuantumBase):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.fusion_norm = nn.LayerNorm(2 * cfg.n_qubits)
        self.classifier = nn.Linear(2 * cfg.n_qubits, len(cfg.class_names))

    def forward(
        self,
        images: torch.Tensor,
        noise_type: str = "none",
        noise_strength: float = 0.0,
        shots: Optional[int] = None,
        compute_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        z_raw, z_angle = self.encode(images)
        q_out = self.quantum_features(z_angle, noise_type, noise_strength, shots)
        diagnostics = self.quantum_diagnostics(
            z_angle,
            q_out,
            noise_type,
            noise_strength,
            shots,
            compute_diagnostics,
        )
        fusion = self.fusion_norm(torch.cat([z_angle, q_out], dim=1))
        return {
            "logits": self.classifier(fusion),
            "z_raw": z_raw,
            "z_angle": z_angle,
            "q_out": q_out,
            **diagnostics,
        }


class HybridAdaptive(QuantumBase):
    """Reliability-guided adaptive fusion using true purity and Z-polarization."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        gate_dim = len(cfg.adaptive_gate_inputs)
        self.gate = nn.Sequential(nn.Linear(gate_dim, 1), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(2 * cfg.n_qubits)
        self.classifier = nn.Linear(2 * cfg.n_qubits, len(cfg.class_names))

    def forward(
        self,
        images: torch.Tensor,
        noise_type: str = "none",
        noise_strength: float = 0.0,
        shots: Optional[int] = None,
        compute_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        z_raw, z_angle = self.encode(images)
        q_out = self.quantum_features(z_angle, noise_type, noise_strength, shots)
        # Adaptive fusion always requires reliability diagnostics. Global purity is
        # analytic even when classification uses finite-shot features.
        diagnostics = self.quantum_diagnostics(
            z_angle,
            q_out,
            noise_type,
            noise_strength,
            shots,
            compute_full=True,
        )
        gate_features = []
        for name in self.cfg.adaptive_gate_inputs:
            if name not in diagnostics:
                raise KeyError(f"Adaptive gate input is unavailable: {name}")
            gate_features.append(diagnostics[name].detach())
        gate_input = torch.stack(gate_features, dim=1)
        alpha = self.gate(gate_input)
        fusion = self.fusion_norm(torch.cat([z_angle, alpha * q_out], dim=1))
        return {
            "logits": self.classifier(fusion),
            "z_raw": z_raw,
            "z_angle": z_angle,
            "q_out": q_out,
            "fusion_gate": alpha.squeeze(1),
            **diagnostics,
        }


def build_model(name: str, cfg: Config) -> BaseClassifier:
    constructors = {
        "classical_full": ClassicalFull,
        "classical_bottleneck": ClassicalBottleneck,
        "classical_mlp": ClassicalMLP,
        "vqc_head": VQCHead,
        "hybrid_fixed": HybridFixed,
        "hybrid_adaptive": HybridAdaptive,
    }
    if name not in constructors:
        raise ValueError(name)
    return constructors[name](cfg)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


# =============================================================================
# Metrics and calibration
# =============================================================================

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        total += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(total)


def multiclass_brier(
    probabilities: np.ndarray, labels: np.ndarray, num_classes: int
) -> float:
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def safe_multiclass_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[labels]
    try:
        return float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        return float("nan")


def safe_macro_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[labels]
    values = []
    for class_index in range(probabilities.shape[1]):
        if one_hot[:, class_index].sum() == 0:
            continue
        values.append(
            average_precision_score(one_hot[:, class_index], probabilities[:, class_index])
        )
    return float(np.mean(values)) if values else float("nan")


def per_class_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    one_hot = np.eye(len(class_names))[labels]
    rows = []
    for index, name in enumerate(class_names):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        try:
            auroc = roc_auc_score(one_hot[:, index], probabilities[:, index])
        except ValueError:
            auroc = float("nan")
        try:
            auprc = average_precision_score(one_hot[:, index], probabilities[:, index])
        except ValueError:
            auprc = float("nan")
        rows.append(
            {
                "class_index": index,
                "class_name": name,
                "precision": float(precision[index]),
                "sensitivity_recall": float(recall[index]),
                "specificity": float(specificity),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "auroc": float(auroc),
                "auprc": float(auprc),
            }
        )
    return rows


def compute_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    ece_bins: int,
) -> Dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities.argmax(axis=1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auroc_ovr": safe_multiclass_auroc(labels, probabilities),
        "auprc_macro": safe_macro_auprc(labels, probabilities),
        "ece": expected_calibration_error(probabilities, labels, ece_bins),
        "brier": multiclass_brier(probabilities, labels, len(class_names)),
        "nll": float(log_loss(labels, clipped, labels=list(range(len(class_names))))),
    }


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(1e-3, 100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    steps: int,
    learning_rate: float,
) -> float:
    scaler = TemperatureScaler()
    tensor_logits = torch.tensor(logits, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    optimizer = optim.LBFGS(
        scaler.parameters(), lr=learning_rate, max_iter=steps, line_search_fn="strong_wolfe"
    )
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(scaler(tensor_logits), tensor_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.detach().item())


# =============================================================================
# Training and prediction
# =============================================================================

def choose_training_noise(
    condition: TrainCondition, rng: random.Random
) -> Tuple[str, float]:
    if condition.noise_type == "mixed":
        noise_type = rng.choice(condition.mixed_noise_types)
        strength = rng.uniform(0.0, condition.mixed_max_strength)
        return noise_type, strength
    return condition.noise_type, condition.noise_strength


def class_weights_from_indices(
    labels: np.ndarray, train_idx: np.ndarray, num_classes: int, device: str
) -> torch.Tensor:
    counts = np.bincount(labels[train_idx], minlength=num_classes).astype(float)
    weights = counts.sum() / np.maximum(counts * num_classes, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]], device: str
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[str]]:
    images, labels, indices, paths = batch
    return (
        images.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        np.asarray(indices, dtype=int),
        list(paths),
    )


def forward_model(
    model: BaseClassifier,
    images: torch.Tensor,
    noise_type: str,
    noise_strength: float,
    shots: Optional[int],
    compute_diagnostics: bool,
) -> Dict[str, torch.Tensor]:
    if model.is_quantum:
        return model(
            images,
            noise_type=noise_type,
            noise_strength=noise_strength,
            shots=shots,
            compute_diagnostics=compute_diagnostics,
        )
    return model(images)


def predict_loader(
    model: BaseClassifier,
    loader: DataLoader,
    cfg: Config,
    noise_type: str = "none",
    noise_strength: float = 0.0,
    shots: Optional[int] = None,
    compute_diagnostics: bool = False,
    temperature: float = 1.0,
) -> Dict[str, np.ndarray]:
    model.eval()
    storage: Dict[str, List[np.ndarray]] = defaultdict(list)
    paths_all: List[str] = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            images, labels, indices, paths = move_batch(batch, cfg.device)
            diagnostics_allowed = compute_diagnostics
            if cfg.diagnostics_max_samples is not None:
                diagnostics_allowed = processed < cfg.diagnostics_max_samples
            output = forward_model(
                model,
                images,
                noise_type,
                noise_strength,
                shots,
                diagnostics_allowed,
            )
            logits = output["logits"]
            probabilities = torch.softmax(logits / temperature, dim=1)
            storage["logits"].append(logits.detach().cpu().numpy())
            storage["probabilities"].append(probabilities.detach().cpu().numpy())
            storage["labels"].append(labels.detach().cpu().numpy())
            storage["sample_index"].append(indices)
            paths_all.extend(paths)
            for key in (
                "global_purity",
                "local_purity",
                "z_polarization",
                "fusion_gate",
            ):
                if key in output:
                    storage[key].append(output[key].detach().cpu().numpy())
            processed += len(labels)
    result = {key: np.concatenate(values, axis=0) for key, values in storage.items()}
    result["paths"] = np.array(paths_all, dtype=object)
    return result


def train_one_epoch(
    model: BaseClassifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    cfg: Config,
    condition: TrainCondition,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler,
) -> Dict[str, float]:
    model.train()
    if epoch < cfg.freeze_backbone_epochs:
        model.set_backbone_trainable(False)
    else:
        model.set_backbone_trainable(True)
    rng = random.Random(cfg.bootstrap_seed + epoch)
    losses = []
    labels_all, probabilities_all = [], []
    for batch in loader:
        images, labels, _, _ = move_batch(batch, cfg.device)
        noise_type, noise_strength = (
            choose_training_noise(condition, rng)
            if model.is_quantum
            else ("none", 0.0)
        )
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = cfg.amp and cfg.device.startswith("cuda") and not model.is_quantum
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = forward_model(
                model,
                images,
                noise_type,
                noise_strength,
                shots=None,
                compute_diagnostics=False,
            )
            loss = criterion(output["logits"], labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        probabilities = torch.softmax(output["logits"].detach(), dim=1)
        losses.append(float(loss.detach().cpu()))
        labels_all.append(labels.detach().cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    metrics = compute_metrics(
        np.concatenate(probabilities_all),
        np.concatenate(labels_all),
        cfg.class_names,
        cfg.ece_bins,
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_experiment(
    model_name: str,
    condition: TrainCondition,
    seed: int,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    labels: np.ndarray,
    train_idx: np.ndarray,
    experiment_dir: Path,
) -> Tuple[BaseClassifier, List[Dict[str, Any]]]:
    seed_everything(seed)
    model = build_model(model_name, cfg).to(cfg.device)
    parameters = count_parameters(model)
    save_json(parameters, experiment_dir / "parameter_counts.json")
    if cfg.class_weighted_loss:
        weights = class_weights_from_indices(
            labels, train_idx, len(cfg.class_names), cfg.device
        )
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.amp and cfg.device.startswith("cuda") and not model.is_quantum
    )
    best_path = experiment_dir / "best_model.pt"
    latest_path = experiment_dir / "checkpoint_latest.pt"
    history: List[Dict[str, Any]] = []
    best_metric = -float("inf")
    start_epoch = 0
    no_improvement = 0

    if cfg.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        history = checkpoint.get("history", [])

    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            cfg,
            condition,
            epoch,
            scaler,
        )
        validation_pack = predict_loader(model, val_loader, cfg)
        validation_metrics = compute_metrics(
            validation_pack["probabilities"],
            validation_pack["labels"],
            cfg.class_names,
            cfg.ece_bins,
        )
        selected = validation_metrics[cfg.selection_metric]
        scheduler.step(selected)
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        pd.DataFrame(history).to_csv(experiment_dir / "history.csv", index=False)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "history": history,
            "model_name": model_name,
            "condition": asdict(condition),
            "seed": seed,
            "config": asdict(cfg),
        }
        torch.save(checkpoint, latest_path)

        if np.isfinite(selected) and selected > best_metric + 1e-5:
            best_metric = float(selected)
            no_improvement = 0
            checkpoint["best_metric"] = best_metric
            torch.save(checkpoint, best_path)
        else:
            no_improvement += 1
        print(
            f"[{model_name}/{condition.name}/seed={seed}] epoch {epoch + 1:02d} "
            f"val_AUROC={validation_metrics['auroc_ovr']:.4f} "
            f"val_F1={validation_metrics['macro_f1']:.4f}"
        )
        if no_improvement >= cfg.early_stopping_patience:
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was produced: {experiment_dir}")
    best = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(best["model"])
    return model, history


# =============================================================================
# Bootstrap, statistical tests, and selective prediction
# =============================================================================

def metric_by_name(
    metric: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
) -> float:
    metrics = compute_metrics(probabilities, labels, cfg.class_names, cfg.ece_bins)
    return float(metrics[metric])


def bootstrap_ci(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metric: str,
    cfg: Config,
    seed_offset: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.bootstrap_seed + seed_offset)
    n = len(labels)
    estimates = []
    for _ in range(cfg.bootstrap_resamples):
        indices = rng.integers(0, n, size=n)
        if len(np.unique(labels[indices])) < len(cfg.class_names):
            continue
        value = metric_by_name(metric, probabilities[indices], labels[indices], cfg)
        if np.isfinite(value):
            estimates.append(value)
    point = metric_by_name(metric, probabilities, labels, cfg)
    if not estimates:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan")}
    lower = 100 * cfg.alpha / 2
    upper = 100 * (1 - cfg.alpha / 2)
    return {
        "estimate": point,
        "ci_low": float(np.percentile(estimates, lower)),
        "ci_high": float(np.percentile(estimates, upper)),
    }


def paired_bootstrap_difference(
    clean_probabilities: np.ndarray,
    noisy_probabilities: np.ndarray,
    labels: np.ndarray,
    metric: str,
    cfg: Config,
    seed_offset: int = 0,
) -> Dict[str, float]:
    if clean_probabilities.shape != noisy_probabilities.shape:
        raise ValueError("Paired bootstrap requires equal prediction shapes.")
    rng = np.random.default_rng(cfg.bootstrap_seed + 100_000 + seed_offset)
    n = len(labels)
    differences = []
    for _ in range(cfg.bootstrap_resamples):
        indices = rng.integers(0, n, size=n)
        if len(np.unique(labels[indices])) < len(cfg.class_names):
            continue
        clean_value = metric_by_name(metric, clean_probabilities[indices], labels[indices], cfg)
        noisy_value = metric_by_name(metric, noisy_probabilities[indices], labels[indices], cfg)
        difference = noisy_value - clean_value
        if np.isfinite(difference):
            differences.append(difference)
    point = metric_by_name(metric, noisy_probabilities, labels, cfg) - metric_by_name(
        metric, clean_probabilities, labels, cfg
    )
    if not differences:
        return {
            "difference": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
    array = np.asarray(differences)
    lower = 100 * cfg.alpha / 2
    upper = 100 * (1 - cfg.alpha / 2)
    p_value = 2 * min(float(np.mean(array <= 0)), float(np.mean(array >= 0)))
    return {
        "difference": float(point),
        "ci_low": float(np.percentile(array, lower)),
        "ci_high": float(np.percentile(array, upper)),
        "p_value": min(1.0, p_value),
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p_values_array = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values_array)
    adjusted = np.full(len(p_values_array), np.nan)
    running = 0.0
    m = len(p_values_array)
    for rank, index in enumerate(order):
        raw = p_values_array[index]
        if not np.isfinite(raw):
            continue
        candidate = min(1.0, (m - rank) * raw)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def risk_coverage_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    score: np.ndarray,
    points: int,
    minimum_samples: int,
) -> pd.DataFrame:
    order = np.argsort(-score)
    coverages = np.linspace(0.05, 1.0, points)
    rows = []
    predictions = probabilities.argmax(axis=1)
    for coverage in coverages:
        count = max(minimum_samples, int(round(coverage * len(labels))))
        count = min(count, len(labels))
        selected = order[:count]
        accuracy = accuracy_score(labels[selected], predictions[selected])
        rows.append(
            {
                "coverage": count / len(labels),
                "selective_risk": 1.0 - accuracy,
                "selective_accuracy": accuracy,
                "n_retained": count,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("coverage")


def area_under_risk_coverage(frame: pd.DataFrame) -> float:
    ordered = frame.sort_values("coverage")
    return float(np.trapz(ordered["selective_risk"], ordered["coverage"]))


def robustness_auc(
    strengths: np.ndarray, metric_values: np.ndarray, normalize: bool = True
) -> float:
    order = np.argsort(strengths)
    x = strengths[order]
    y = metric_values[order]
    if normalize and y[0] != 0:
        y = y / y[0]
    denominator = x[-1] - x[0]
    if denominator <= 0:
        return float("nan")
    return float(np.trapz(y, x) / denominator)


# =============================================================================
# Plotting
# =============================================================================

def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_history(history: List[Dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(history)
    if frame.empty:
        return
    for metric in ("auroc_ovr", "macro_f1", "nll", "ece", "loss"):
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        if train_col not in frame and val_col not in frame:
            continue
        plt.figure(figsize=(7, 5))
        if train_col in frame:
            plt.plot(frame["epoch"], frame[train_col], marker="o", label="Training")
        if val_col in frame:
            plt.plot(frame["epoch"], frame[val_col], marker="o", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(alpha=0.3)
        save_figure(output_dir / f"history_{metric}.png")


def plot_confusion(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> None:
    matrix = confusion_matrix(labels, probabilities.argmax(axis=1))
    plt.figure(figsize=(6, 5))
    image = plt.imshow(matrix, interpolation="nearest")
    plt.colorbar(image)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right")
    plt.yticks(ticks, class_names)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    save_figure(path)


def plot_roc_pr(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
) -> None:
    one_hot = np.eye(len(class_names))[labels]
    plt.figure(figsize=(7, 5))
    for index, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(one_hot[:, index], probabilities[:, index])
        area = roc_auc_score(one_hot[:, index], probabilities[:, index])
        plt.plot(fpr, tpr, label=f"{name} (AUC={area:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(output_dir / "roc_curves.png")

    plt.figure(figsize=(7, 5))
    for index, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            one_hot[:, index], probabilities[:, index]
        )
        area = average_precision_score(one_hot[:, index], probabilities[:, index])
        plt.plot(recall, precision, label=f"{name} (AP={area:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(output_dir / "precision_recall_curves.png")


def plot_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int,
    path: Path,
) -> None:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    centers, accuracies, confidences = [], [], []
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if not np.any(mask):
            continue
        centers.append((edges[index] + edges[index + 1]) / 2)
        accuracies.append(correctness[mask].mean())
        confidences.append(confidence[mask].mean())
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(confidences, accuracies, marker="o", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(path)


def plot_risk_coverage(curves: Mapping[str, pd.DataFrame], path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for name, frame in curves.items():
        plt.plot(
            frame["coverage"], frame["selective_risk"], marker="o", label=name
        )
    plt.xlabel("Coverage")
    plt.ylabel("Selective risk")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(path)


def plot_noise_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        return
    analytic = frame[frame["shots"].astype(str) == "analytic"].copy()
    for metric in (
        "auroc_ovr",
        "macro_f1",
        "ece",
        "global_purity_mean",
        "local_purity_mean",
        "z_polarization_mean",
    ):
        if metric not in analytic.columns:
            continue
        plt.figure(figsize=(8, 5))
        for noise_type, subset in analytic.groupby("eval_noise_type"):
            grouped = subset.groupby("noise_strength")[metric]
            means = grouped.mean()
            standard_errors = grouped.std(ddof=1) / np.sqrt(grouped.count())
            plt.plot(means.index, means.values, marker="o", label=noise_type)
            if len(subset["seed"].unique()) > 1:
                plt.fill_between(
                    means.index,
                    means.values - 1.96 * standard_errors.values,
                    means.values + 1.96 * standard_errors.values,
                    alpha=0.2,
                )
        plt.xlabel("Noise strength")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(alpha=0.3)
        save_figure(output_dir / f"{metric}_vs_noise.png")

    if {"global_purity_mean", "auroc_ovr"}.issubset(analytic.columns):
        plt.figure(figsize=(7, 5))
        for noise_type, subset in analytic.groupby("eval_noise_type"):
            plt.plot(
                subset["global_purity_mean"],
                subset["auroc_ovr"],
                marker="o",
                linestyle="none",
                label=noise_type,
            )
        plt.xlabel("True global purity Tr(rho²)")
        plt.ylabel("AUROC")
        plt.legend()
        plt.grid(alpha=0.3)
        save_figure(output_dir / "auroc_vs_true_global_purity.png")


# =============================================================================
# Experiment evaluation and aggregation
# =============================================================================

def pack_to_prediction_frame(
    pack: Dict[str, np.ndarray],
    cfg: Config,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    probabilities = pack["probabilities"]
    logits = pack["logits"]
    frame = pd.DataFrame(
        {
            "sample_index": pack["sample_index"].astype(int),
            "path": pack["paths"],
            "true_label_index": pack["labels"].astype(int),
            "true_label": [cfg.class_names[index] for index in pack["labels"].astype(int)],
            "predicted_label_index": probabilities.argmax(axis=1),
            "predicted_label": [
                cfg.class_names[index] for index in probabilities.argmax(axis=1)
            ],
            "confidence": probabilities.max(axis=1),
            "predictive_entropy": predictive_entropy(probabilities),
        }
    )
    for index, name in enumerate(cfg.class_names):
        frame[f"logit_{name}"] = logits[:, index]
        frame[f"probability_{name}"] = probabilities[:, index]
    for key in ("global_purity", "local_purity", "z_polarization", "fusion_gate"):
        if key in pack and len(pack[key]) == len(frame):
            frame[key] = pack[key]
    for key, value in metadata.items():
        frame[key] = value
    return frame


def evaluate_single_condition(
    model: BaseClassifier,
    loader: DataLoader,
    cfg: Config,
    experiment_dir: Path,
    model_name: str,
    train_condition: TrainCondition,
    seed: int,
    split_name: str,
    eval_noise_type: str,
    noise_strength: float,
    shots: Optional[int],
    temperature: float,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], pd.DataFrame]:
    compute_diagnostics = model.is_quantum and cfg.compute_quantum_diagnostics
    pack = predict_loader(
        model,
        loader,
        cfg,
        noise_type=eval_noise_type if model.is_quantum else "none",
        noise_strength=noise_strength if model.is_quantum else 0.0,
        shots=shots if model.is_quantum else None,
        compute_diagnostics=compute_diagnostics,
        temperature=temperature,
    )
    metrics = compute_metrics(
        pack["probabilities"], pack["labels"], cfg.class_names, cfg.ece_bins
    )
    row: Dict[str, Any] = {
        "model": model_name,
        "train_condition": train_condition.name,
        "train_noise_type": train_condition.noise_type,
        "train_noise_strength": train_condition.noise_strength,
        "seed": seed,
        "split": split_name,
        "eval_noise_type": eval_noise_type if model.is_quantum else "none",
        "noise_strength": noise_strength if model.is_quantum else 0.0,
        "shots": "analytic" if shots is None else int(shots),
        "temperature": temperature,
        **metrics,
    }
    for key in ("global_purity", "local_purity", "z_polarization", "fusion_gate"):
        if key in pack:
            row[f"{key}_mean"] = float(np.mean(pack[key]))
            row[f"{key}_std"] = float(np.std(pack[key], ddof=1))
    metadata = {
        "model": model_name,
        "train_condition": train_condition.name,
        "seed": seed,
        "split": split_name,
        "eval_noise_type": row["eval_noise_type"],
        "noise_strength": row["noise_strength"],
        "shots": row["shots"],
        "temperature": temperature,
    }
    prediction_frame = pack_to_prediction_frame(pack, cfg, metadata)
    return row, pack, prediction_frame


def save_clean_diagnostics(
    pack: Dict[str, np.ndarray],
    cfg: Config,
    output_dir: Path,
    prefix: str,
) -> None:
    metrics = compute_metrics(
        pack["probabilities"], pack["labels"], cfg.class_names, cfg.ece_bins
    )
    save_json(metrics, output_dir / f"{prefix}_metrics.json")
    pd.DataFrame(per_class_metrics(pack["probabilities"], pack["labels"], cfg.class_names)).to_csv(
        output_dir / f"{prefix}_per_class_metrics.csv", index=False
    )
    plot_confusion(
        pack["probabilities"],
        pack["labels"],
        cfg.class_names,
        output_dir / f"{prefix}_confusion_matrix.png",
    )
    plot_roc_pr(pack["probabilities"], pack["labels"], cfg.class_names, output_dir / prefix)
    plot_reliability_diagram(
        pack["probabilities"],
        pack["labels"],
        cfg.ece_bins,
        output_dir / f"{prefix}_reliability_diagram.png",
    )


def selective_analysis(
    pack: Dict[str, np.ndarray], cfg: Config, output_dir: Path
) -> None:
    probabilities = pack["probabilities"]
    labels = pack["labels"]
    score_map: Dict[str, np.ndarray] = {
        "confidence": probabilities.max(axis=1),
        "negative_entropy": -predictive_entropy(probabilities),
    }
    for key in ("global_purity", "local_purity", "z_polarization"):
        if key in pack and len(pack[key]) == len(labels):
            score_map[key] = pack[key]
    curves = {}
    summaries = []
    for name, score in score_map.items():
        curve = risk_coverage_curve(
            probabilities,
            labels,
            score,
            cfg.risk_coverage_points,
            cfg.minimum_selective_samples,
        )
        curve["selection_score"] = name
        curve.to_csv(output_dir / f"risk_coverage_{name}.csv", index=False)
        curves[name] = curve
        summaries.append(
            {"selection_score": name, "aurc": area_under_risk_coverage(curve)}
        )
    pd.DataFrame(summaries).to_csv(output_dir / "risk_coverage_summary.csv", index=False)
    plot_risk_coverage(curves, output_dir / "risk_coverage_comparison.png")


def bootstrap_analysis(
    condition_packs: Mapping[Tuple[str, float, str], Dict[str, np.ndarray]],
    cfg: Config,
    output_dir: Path,
) -> None:
    rows = []
    paired_rows = []
    for condition_key, pack in condition_packs.items():
        noise_type, strength, shots = condition_key
        for metric_index, metric in enumerate(cfg.bootstrap_metrics):
            result = bootstrap_ci(
                pack["probabilities"],
                pack["labels"],
                metric,
                cfg,
                seed_offset=metric_index,
            )
            rows.append(
                {
                    "eval_noise_type": noise_type,
                    "noise_strength": strength,
                    "shots": shots,
                    "metric": metric,
                    **result,
                }
            )

    analytic_keys = [key for key in condition_packs if key[2] == "analytic"]
    for noise_type in sorted({key[0] for key in analytic_keys}):
        clean_key = (noise_type, 0.0, "analytic")
        if clean_key not in condition_packs:
            continue
        clean_pack = condition_packs[clean_key]
        for key in analytic_keys:
            current_type, strength, shots = key
            if current_type != noise_type or strength == 0:
                continue
            noisy_pack = condition_packs[key]
            if not np.array_equal(clean_pack["sample_index"], noisy_pack["sample_index"]):
                raise ValueError("Prediction order changed; paired bootstrap is invalid.")
            for metric_index, metric in enumerate(cfg.bootstrap_metrics):
                result = paired_bootstrap_difference(
                    clean_pack["probabilities"],
                    noisy_pack["probabilities"],
                    clean_pack["labels"],
                    metric,
                    cfg,
                    seed_offset=metric_index + int(strength * 10_000),
                )
                paired_rows.append(
                    {
                        "eval_noise_type": noise_type,
                        "noise_strength": strength,
                        "metric": metric,
                        **result,
                    }
                )
    if paired_rows:
        adjusted = holm_adjust([row["p_value"] for row in paired_rows])
        for row, value in zip(paired_rows, adjusted):
            row["p_holm"] = value
    pd.DataFrame(rows).to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "paired_bootstrap_differences.csv", index=False
    )


def correlation_and_robustness_analysis(
    metrics_frame: pd.DataFrame, output_dir: Path
) -> None:
    analytic = metrics_frame[metrics_frame["shots"].astype(str) == "analytic"].copy()
    correlation_rows = []
    robustness_rows = []
    for noise_type, subset in analytic.groupby("eval_noise_type"):
        subset = subset.sort_values("noise_strength")
        for diagnostic in (
            "global_purity_mean",
            "local_purity_mean",
            "z_polarization_mean",
        ):
            if diagnostic not in subset or subset[diagnostic].notna().sum() < 3:
                continue
            x = subset[diagnostic].to_numpy(dtype=float)
            y = subset["auroc_ovr"].to_numpy(dtype=float)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            coefficients = np.polyfit(x, y, deg=2)
            predicted = np.polyval(coefficients, x)
            denominator = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - np.sum((y - predicted) ** 2) / denominator if denominator else np.nan
            correlation_rows.append(
                {
                    "eval_noise_type": noise_type,
                    "diagnostic": diagnostic,
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_rho": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                    "quadratic_r2": r_squared,
                    "quadratic_coefficients": json.dumps(coefficients.tolist()),
                    "n_points": len(subset),
                }
            )
        for metric in ("auroc_ovr", "macro_f1", "auprc_macro", "ece"):
            robustness_rows.append(
                {
                    "eval_noise_type": noise_type,
                    "metric": metric,
                    "robustness_auc": robustness_auc(
                        subset["noise_strength"].to_numpy(dtype=float),
                        subset[metric].to_numpy(dtype=float),
                        normalize=metric != "ece",
                    ),
                }
            )
    pd.DataFrame(correlation_rows).to_csv(
        output_dir / "purity_performance_correlations.csv", index=False
    )
    pd.DataFrame(robustness_rows).to_csv(
        output_dir / "robustness_area_under_curve.csv", index=False
    )


def evaluate_experiment(
    model: BaseClassifier,
    model_name: str,
    condition: TrainCondition,
    seed: int,
    cfg: Config,
    val_loader: DataLoader,
    test_loader: DataLoader,
    experiment_dir: Path,
) -> pd.DataFrame:
    validation_clean = predict_loader(model, val_loader, cfg)
    temperature = 1.0
    if cfg.temperature_scaling:
        temperature = fit_temperature(
            validation_clean["logits"],
            validation_clean["labels"],
            cfg.temperature_steps,
            cfg.temperature_lr,
        )
    save_json({"temperature": temperature}, experiment_dir / "temperature.json")

    rows = []
    condition_packs: Dict[Tuple[str, float, str], Dict[str, np.ndarray]] = {}
    predictions_dir = experiment_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if not model.is_quantum:
        row, pack, prediction_frame = evaluate_single_condition(
            model,
            test_loader,
            cfg,
            experiment_dir,
            model_name,
            condition,
            seed,
            "test",
            "none",
            0.0,
            None,
            temperature,
        )
        rows.append(row)
        prediction_frame.to_csv(predictions_dir / "test_predictions.csv", index=False)
        save_clean_diagnostics(pack, cfg, experiment_dir, "test")
        selective_analysis(pack, cfg, experiment_dir / "selective_prediction")
        condition_packs[("none", 0.0, "analytic")] = pack
        bootstrap_analysis(condition_packs, cfg, experiment_dir / "statistics")
    else:
        for noise_type in cfg.eval_noise_types:
            for strength in cfg.noise_strengths:
                for shots in cfg.shot_settings:
                    shot_label = "analytic" if shots is None else str(shots)
                    row, pack, prediction_frame = evaluate_single_condition(
                        model,
                        test_loader,
                        cfg,
                        experiment_dir,
                        model_name,
                        condition,
                        seed,
                        "test",
                        noise_type,
                        strength,
                        shots,
                        temperature,
                    )
                    rows.append(row)
                    condition_packs[(noise_type, float(strength), shot_label)] = pack
                    prediction_frame.to_csv(
                        predictions_dir
                        / f"test_{noise_type}_{strength:.3f}_{shot_label}.csv",
                        index=False,
                    )
                    if strength == 0 and shots is None and noise_type == cfg.eval_noise_types[0]:
                        save_clean_diagnostics(pack, cfg, experiment_dir, "test_clean")
                        selective_analysis(pack, cfg, experiment_dir / "selective_prediction")
        bootstrap_analysis(condition_packs, cfg, experiment_dir / "statistics")

    frame = pd.DataFrame(rows)
    frame.to_csv(experiment_dir / "evaluation_metrics.csv", index=False)
    correlation_and_robustness_analysis(frame, experiment_dir / "statistics")
    plot_noise_summary(frame, experiment_dir / "figures")
    return frame


def experiment_specs(cfg: Config) -> List[Tuple[str, TrainCondition]]:
    clean = TrainCondition(name="clean", noise_type="none", noise_strength=0.0)
    specs = []
    for model_name in cfg.model_variants:
        if model_name in QUANTUM_MODEL_NAMES:
            for condition in cfg.train_conditions:
                specs.append((model_name, condition))
        else:
            specs.append((model_name, clean))
    return specs


def aggregate_across_seeds(all_metrics: pd.DataFrame, output_dir: Path) -> None:
    if all_metrics.empty:
        return
    group_columns = [
        "model",
        "train_condition",
        "eval_noise_type",
        "noise_strength",
        "shots",
    ]
    numeric_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "auroc_ovr",
        "auprc_macro",
        "ece",
        "brier",
        "nll",
        "global_purity_mean",
        "local_purity_mean",
        "z_polarization_mean",
        "fusion_gate_mean",
    ]
    rows = []
    for keys, subset in all_metrics.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        n = subset["seed"].nunique()
        row["n_seeds"] = n
        for metric in numeric_metrics:
            if metric not in subset:
                continue
            values = subset[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            critical = t.ppf(0.975, df=len(values) - 1) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = mean - critical * sem
            row[f"{metric}_ci_high"] = mean + critical * sem
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(output_dir / "multi_seed_summary.csv", index=False)
    plot_noise_summary(all_metrics, output_dir / "figures_multi_seed")


def cross_channel_matrix(all_metrics: pd.DataFrame, output_dir: Path) -> None:
    analytic = all_metrics[
        (all_metrics["shots"].astype(str) == "analytic")
        & (all_metrics["noise_strength"] == 0.05)
        & (all_metrics["model"].isin(["hybrid_fixed", "hybrid_adaptive"]))
    ]
    if analytic.empty:
        return
    grouped = (
        analytic.groupby(["model", "train_condition", "eval_noise_type"])["auroc_ovr"]
        .mean()
        .reset_index()
    )
    grouped.to_csv(output_dir / "cross_channel_auroc_at_gamma_005.csv", index=False)
    for model_name, subset in grouped.groupby("model"):
        pivot = subset.pivot(
            index="train_condition", columns="eval_noise_type", values="auroc_ovr"
        )
        plt.figure(figsize=(7, 5))
        image = plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(image, label="Mean AUROC")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                value = pivot.iloc[row, column]
                if np.isfinite(value):
                    plt.text(column, row, f"{value:.3f}", ha="center", va="center")
        plt.xlabel("Evaluation noise channel")
        plt.ylabel("Training condition")
        save_figure(output_dir / f"cross_channel_{model_name}.png")


# =============================================================================
# Main orchestration
# =============================================================================

def run_pipeline(cfg: Config) -> None:
    cfg.validate()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_json(asdict(cfg), root / "resolved_config.json")
    save_json(capture_environment(), root / "environment.json")

    _, eval_transform = make_transforms(cfg)
    _, labels, groups, paths, class_to_idx = load_manifest_or_imagefolder(
        cfg, eval_transform
    )
    train_idx, val_idx, test_idx = create_or_load_splits(
        cfg, labels, groups, root
    )
    dataset_audit(
        cfg,
        paths,
        labels,
        groups,
        train_idx,
        val_idx,
        test_idx,
        root,
    )
    split_summary = {
        "n_total": len(labels),
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "class_to_idx": class_to_idx,
        "train_class_counts": class_distribution(labels, train_idx, cfg.class_names),
        "validation_class_counts": class_distribution(labels, val_idx, cfg.class_names),
        "test_class_counts": class_distribution(labels, test_idx, cfg.class_names),
        "patient_level_split": cfg.patient_level_split,
    }
    save_json(split_summary, root / "split_summary.json")

    all_metric_frames = []
    specs = experiment_specs(cfg)
    for seed in cfg.seeds:
        cfg.quantum_device_seed = seed
        train_loader, val_loader, test_loader, labels_current, _, _, _ = build_loaders(
            cfg, train_idx, val_idx, test_idx, seed
        )
        if not np.array_equal(labels, labels_current):
            raise RuntimeError("Dataset labels changed between runs.")
        for model_name, condition in specs:
            experiment_name = f"{model_name}__{condition.name}__seed_{seed}"
            experiment_dir = root / "experiments" / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = experiment_dir / "evaluation_metrics.csv"
            if metrics_path.exists() and not cfg.overwrite:
                print(f"[Skip existing] {experiment_name}")
                frame = pd.read_csv(metrics_path)
                all_metric_frames.append(frame)
                continue
            save_json(
                {
                    "model_name": model_name,
                    "training_condition": asdict(condition),
                    "seed": seed,
                },
                experiment_dir / "experiment_spec.json",
            )
            model, history = train_experiment(
                model_name,
                condition,
                seed,
                cfg,
                train_loader,
                val_loader,
                labels,
                train_idx,
                experiment_dir,
            )
            plot_training_history(history, experiment_dir / "figures")
            frame = evaluate_experiment(
                model,
                model_name,
                condition,
                seed,
                cfg,
                val_loader,
                test_loader,
                experiment_dir,
            )
            all_metric_frames.append(frame)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_metrics = pd.concat(all_metric_frames, ignore_index=True)
    all_metrics.to_csv(root / "all_experiment_metrics.csv", index=False)
    aggregate_across_seeds(all_metrics, root / "summary")
    cross_channel_matrix(all_metrics, root / "summary")
    create_manuscript_tables(all_metrics, cfg, root / "summary")
    print(f"Completed. Outputs are in: {root}")


def create_manuscript_tables(
    all_metrics: pd.DataFrame, cfg: Config, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    analytic = all_metrics[all_metrics["shots"].astype(str) == "analytic"].copy()
    clean = analytic[analytic["noise_strength"] == 0.0]
    clean_summary = (
        clean.groupby(["model", "train_condition"])[
            ["accuracy", "macro_f1", "auroc_ovr", "auprc_macro", "ece", "brier", "nll"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    clean_summary.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in clean_summary.columns
    ]
    clean_summary.to_csv(output_dir / "table_clean_performance.csv", index=False)

    if not analytic.empty:
        robustness = (
            analytic.groupby(
                ["model", "train_condition", "eval_noise_type", "noise_strength"]
            )[["auroc_ovr", "macro_f1", "ece", "global_purity_mean", "z_polarization_mean"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        robustness.columns = [
            "_".join(str(part) for part in column if part).strip("_")
            if isinstance(column, tuple)
            else str(column)
            for column in robustness.columns
        ]
        robustness.to_csv(output_dir / "table_noise_robustness.csv", index=False)

    finite = all_metrics[all_metrics["shots"].astype(str) != "analytic"]
    if not finite.empty:
        finite.groupby(
            ["model", "train_condition", "shots", "eval_noise_type", "noise_strength"]
        )[["auroc_ovr", "macro_f1", "ece"]].mean().reset_index().to_csv(
            output_dir / "table_finite_shot_results.csv", index=False
        )


# =============================================================================
# Command-line interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON configuration.")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run one seed, one epoch, two models, and a short noise grid.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute existing experiment outputs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.quick_test:
        cfg.quick_test = True
    if args.overwrite:
        cfg.overwrite = True
    cfg.validate()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
