#!/usr/bin/env python3
"""Compare original PyTorch policy outputs with TorchScript and ONNX exports."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_policy import load_reference_policy


def generate_inputs(obs_path: Path, obs_dim: int, num_samples: int | None) -> torch.Tensor:
    lines = obs_path.read_text().strip().splitlines()
    if num_samples is not None:
        lines = lines[:num_samples]
    obs_list = []
    for line in lines:
        values = [float(x) for x in line.strip().strip('[]').split(',')]
        if len(values) < obs_dim:
            raise ValueError(f"Observation entry has {len(values)} values (< {obs_dim})")
        obs_list.append(values[:obs_dim])
    arr = torch.tensor(np.array(obs_list, dtype=np.float32))
    return arr


def compare_models(checkpoint: Path, ts_path: Path, onnx_path: Path, obs_path: Path, num_samples: int | None) -> dict:
    with torch.no_grad():
        ref_policy, running_mean, _, _ = load_reference_policy(checkpoint)
        obs = generate_inputs(obs_path, int(running_mean.numel()), num_samples)
        ref_mean, ref_value, ref_log_std = ref_policy(obs)

        ts_module = torch.jit.load(ts_path)
        ts_mean, ts_value, ts_log_std = ts_module(obs)

    ort_sess = ort.InferenceSession(str(onnx_path))
    ort_outputs = ort_sess.run(None, {"obs": obs.numpy()})
    ort_mean, ort_value, ort_log_std = [torch.from_numpy(arr) for arr in ort_outputs]

    def metrics(ref, other):
        diff = other - ref
        return {
            "max_abs": float(diff.abs().max()),
            "mean_abs": float(diff.abs().mean()),
            "rmse": float(torch.sqrt((diff ** 2).mean())),
        }

    return {
        "torchscript_vs_ref": {
            "mean": metrics(ref_mean, ts_mean),
            "value": metrics(ref_value, ts_value),
            "log_std": metrics(ref_log_std, ts_log_std),
        },
        "onnx_vs_ref": {
            "mean": metrics(ref_mean, ort_mean),
            "value": metrics(ref_value, ort_value),
            "log_std": metrics(ref_log_std, ort_log_std),
        },
        "num_samples": obs.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate exported policy against PyTorch reference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_agent.pt checkpoint")
    parser.add_argument("--torchscript", type=Path, default=Path("deployment/policy/harold_policy.ts"))
    parser.add_argument("--onnx", type=Path, default=Path("deployment/policy/harold_policy.onnx"))
    parser.add_argument("--observations", type=Path, default=Path("simulation_logs/observations.log"))
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples from log to compare")
    parser.add_argument("--output", type=Path, default=Path("deployment/policy/export_validation.json"))
    args = parser.parse_args()

    metrics = compare_models(args.checkpoint, args.torchscript, args.onnx, args.observations, args.num_samples)
    print(json.dumps(metrics, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"Validation metrics written to {args.output}")


if __name__ == "__main__":
    main()
