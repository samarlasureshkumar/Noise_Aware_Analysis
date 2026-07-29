#!/usr/bin/env python3
"""
Optional Qiskit Aer/IBM-backend extension for qecv_publication_pipeline.py.

This script evaluates a trained VQC-head or fixed-fusion hybrid checkpoint using:
1. an AerSimulator constructed from a live IBM backend calibration, or
2. a configurable generic Aer noise model with depolarizing, thermal-relaxation,
   and readout errors.

It preserves the trained DenseNet projection, quantum parameters, fusion layer,
and classifier. The variational circuit is executed in Qiskit with finite shots,
and the measured Pauli-Z expectations are returned to the PyTorch classifier.

Examples
--------
Live IBM-derived simulator:
python qecv_qiskit_backend_extension.py \
  --config qecv_publication_config.json \
  --checkpoint /path/hybrid_fixed__clean__seed_11/best_model.pt \
  --model hybrid_fixed --backend-name ibm_brisbane --shots 4096

Generic Aer noise model:
python qecv_qiskit_backend_extension.py \
  --config qecv_publication_config.json \
  --checkpoint /path/hybrid_fixed__clean__seed_11/best_model.pt \
  --model hybrid_fixed --shots 4096 --generic-noise
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from qecv_publication_pipeline import (
    Config,
    HybridFixed,
    VQCHead,
    build_loaders,
    compute_metrics,
    create_or_load_splits,
    load_config,
    load_manifest_or_imagefolder,
    make_transforms,
    per_class_metrics,
    save_json,
    seed_everything,
)


def optional_qiskit_imports():
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import (
            NoiseModel,
            ReadoutError,
            depolarizing_error,
            thermal_relaxation_error,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install qiskit and qiskit-aer before running this extension."
        ) from exc
    return {
        "QuantumCircuit": QuantumCircuit,
        "transpile": transpile,
        "AerSimulator": AerSimulator,
        "NoiseModel": NoiseModel,
        "ReadoutError": ReadoutError,
        "depolarizing_error": depolarizing_error,
        "thermal_relaxation_error": thermal_relaxation_error,
    }


def build_generic_noise_model(
    one_qubit_depolarizing: float,
    two_qubit_depolarizing: float,
    readout_error: float,
    t1_microseconds: float,
    t2_microseconds: float,
    one_qubit_gate_ns: float,
    two_qubit_gate_ns: float,
):
    q = optional_qiskit_imports()
    model = q["NoiseModel"]()
    one_dep = q["depolarizing_error"](one_qubit_depolarizing, 1)
    two_dep = q["depolarizing_error"](two_qubit_depolarizing, 2)
    t1 = t1_microseconds * 1e-6
    t2 = min(t2_microseconds * 1e-6, 2 * t1)
    one_thermal = q["thermal_relaxation_error"](t1, t2, one_qubit_gate_ns * 1e-9)
    thermal_a = q["thermal_relaxation_error"](t1, t2, two_qubit_gate_ns * 1e-9)
    thermal_b = q["thermal_relaxation_error"](t1, t2, two_qubit_gate_ns * 1e-9)
    two_thermal = thermal_a.tensor(thermal_b)
    model.add_all_qubit_quantum_error(one_dep.compose(one_thermal), ["ry", "sx", "x"])
    model.add_all_qubit_quantum_error(two_dep.compose(two_thermal), ["cx"])
    readout = q["ReadoutError"](
        [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
    )
    model.add_all_qubit_readout_error(readout)
    return model


def build_simulator(args: argparse.Namespace):
    q = optional_qiskit_imports()
    if args.backend_name:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as exc:
            raise RuntimeError(
                "Install qiskit-ibm-runtime to use a live IBM backend calibration."
            ) from exc
        token = args.ibm_token or os.getenv("QISKIT_IBM_TOKEN")
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        else:
            service = QiskitRuntimeService()
        backend = service.backend(args.backend_name)
        simulator = q["AerSimulator"].from_backend(backend)
        metadata = {
            "source": "ibm_backend_calibration",
            "backend_name": args.backend_name,
            "backend_version": getattr(backend, "backend_version", None),
        }
        return simulator, metadata

    noise_model = build_generic_noise_model(
        args.one_qubit_depolarizing,
        args.two_qubit_depolarizing,
        args.readout_error,
        args.t1_microseconds,
        args.t2_microseconds,
        args.one_qubit_gate_ns,
        args.two_qubit_gate_ns,
    )
    simulator = q["AerSimulator"](
        noise_model=noise_model,
        seed_simulator=args.seed,
        method="density_matrix",
    )
    metadata = {
        "source": "generic_aer_noise_model",
        "one_qubit_depolarizing": args.one_qubit_depolarizing,
        "two_qubit_depolarizing": args.two_qubit_depolarizing,
        "readout_error": args.readout_error,
        "t1_microseconds": args.t1_microseconds,
        "t2_microseconds": args.t2_microseconds,
        "one_qubit_gate_ns": args.one_qubit_gate_ns,
        "two_qubit_gate_ns": args.two_qubit_gate_ns,
    }
    return simulator, metadata


def build_measured_circuit(
    angles: np.ndarray, weights: np.ndarray, n_qubits: int, q_layers: int
):
    q = optional_qiskit_imports()
    circuit = q["QuantumCircuit"](n_qubits, n_qubits)
    for wire in range(n_qubits):
        circuit.ry(float(angles[wire]), wire)
    index = 0
    for _ in range(q_layers):
        for wire in range(n_qubits):
            circuit.ry(float(weights[index]), wire)
            index += 1
        for wire in range(n_qubits):
            circuit.cx(wire, (wire + 1) % n_qubits)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def counts_to_z_expectations(counts: Dict[str, int], n_qubits: int) -> np.ndarray:
    total = sum(counts.values())
    expectations = np.zeros(n_qubits, dtype=np.float32)
    if total == 0:
        return expectations
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        qiskit_qubit_order = clean[::-1]
        for wire in range(n_qubits):
            eigenvalue = 1.0 if qiskit_qubit_order[wire] == "0" else -1.0
            expectations[wire] += count * eigenvalue
    return expectations / total


def load_trained_model(model_name: str, cfg: Config, checkpoint_path: str):
    if model_name == "hybrid_fixed":
        model = HybridFixed(cfg)
    elif model_name == "vqc_head":
        model = VQCHead(cfg)
    else:
        raise ValueError("Qiskit extension supports hybrid_fixed or vqc_head.")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(cfg.device).eval()
    return model


def classify_from_qiskit_features(model, z_angle: torch.Tensor, q_out: torch.Tensor):
    if isinstance(model, HybridFixed):
        fusion = model.fusion_norm(torch.cat([z_angle, q_out], dim=1))
        return model.classifier(fusion)
    if isinstance(model, VQCHead):
        return model.classifier(q_out)
    raise TypeError(type(model))


def run_backend_evaluation(args: argparse.Namespace) -> None:
    q = optional_qiskit_imports()
    cfg = load_config(args.config)
    cfg.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    seed_everything(args.seed)
    cfg.quantum_device_seed = args.seed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, eval_transform = make_transforms(cfg)
    _, labels, groups, _, _ = load_manifest_or_imagefolder(cfg, eval_transform)
    split_root = Path(cfg.output_dir)
    train_idx, val_idx, test_idx = create_or_load_splits(
        cfg, labels, groups, split_root
    )
    _, _, test_loader, _, _, _, _ = build_loaders(
        cfg, train_idx, val_idx, test_idx, args.seed
    )
    model = load_trained_model(args.model, cfg, args.checkpoint)
    simulator, simulator_metadata = build_simulator(args)

    all_logits: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []
    all_paths: List[str] = []
    all_z_polarization: List[np.ndarray] = []
    q_weights = model.q_weights.detach().cpu().numpy()

    with torch.no_grad():
        for images, batch_labels, indices, paths in test_loader:
            images = images.to(cfg.device)
            _, z_angle = model.encode(images)
            circuits = [
                build_measured_circuit(
                    row.detach().cpu().numpy(), q_weights, cfg.n_qubits, cfg.q_layers
                )
                for row in z_angle
            ]
            transpiled = q["transpile"](
                circuits,
                simulator,
                optimization_level=args.optimization_level,
                seed_transpiler=args.seed,
            )
            result = simulator.run(
                transpiled, shots=args.shots, seed_simulator=args.seed
            ).result()
            q_features = []
            for circuit_index in range(len(transpiled)):
                counts = result.get_counts(circuit_index)
                q_features.append(counts_to_z_expectations(counts, cfg.n_qubits))
            q_out = torch.tensor(
                np.stack(q_features), dtype=torch.float32, device=cfg.device
            )
            logits = classify_from_qiskit_features(model, z_angle, q_out)
            probabilities = torch.softmax(logits / args.temperature, dim=1)
            all_logits.append(logits.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            all_indices.append(np.asarray(indices, dtype=int))
            all_paths.extend(list(paths))
            all_z_polarization.append(q_out.square().mean(dim=1).cpu().numpy())

    logits = np.concatenate(all_logits)
    probabilities = np.concatenate(all_probabilities)
    true_labels = np.concatenate(all_labels)
    sample_indices = np.concatenate(all_indices)
    z_polarization = np.concatenate(all_z_polarization)
    metrics = compute_metrics(
        probabilities, true_labels, cfg.class_names, cfg.ece_bins
    )
    save_json(metrics, output_dir / "backend_derived_metrics.json")
    save_json(simulator_metadata, output_dir / "backend_noise_metadata.json")
    pd.DataFrame(per_class_metrics(probabilities, true_labels, cfg.class_names)).to_csv(
        output_dir / "backend_derived_per_class_metrics.csv", index=False
    )

    frame = pd.DataFrame(
        {
            "sample_index": sample_indices,
            "path": all_paths,
            "true_label_index": true_labels,
            "predicted_label_index": probabilities.argmax(axis=1),
            "confidence": probabilities.max(axis=1),
            "z_polarization": z_polarization,
        }
    )
    for index, name in enumerate(cfg.class_names):
        frame[f"logit_{name}"] = logits[:, index]
        frame[f"probability_{name}"] = probabilities[:, index]
    frame.to_csv(output_dir / "backend_derived_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Outputs written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["hybrid_fixed", "vqc_head"], required=True)
    parser.add_argument("--output-dir", default="./qiskit_backend_evaluation")
    parser.add_argument("--backend-name", default=None)
    parser.add_argument("--ibm-token", default=None)
    parser.add_argument("--generic-noise", action="store_true")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--one-qubit-depolarizing", type=float, default=0.001)
    parser.add_argument("--two-qubit-depolarizing", type=float, default=0.01)
    parser.add_argument("--readout-error", type=float, default=0.02)
    parser.add_argument("--t1-microseconds", type=float, default=120.0)
    parser.add_argument("--t2-microseconds", type=float, default=90.0)
    parser.add_argument("--one-qubit-gate-ns", type=float, default=35.0)
    parser.add_argument("--two-qubit-gate-ns", type=float, default=300.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_backend_evaluation(parse_args())
