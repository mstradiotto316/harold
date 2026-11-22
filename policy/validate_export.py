#!/usr/bin/env python3
"""Compare original PyTorch policy outputs with TorchScript and ONNX exports."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from export_policy import _candidate_run_dirs, load_reference_policy


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


def reference_outputs(checkpoint: Path, obs: torch.Tensor):
    """Use the original skrl model factory when the run preserved its training config."""
    for run_dir in _candidate_run_dirs(checkpoint):
        agent_path = run_dir / "params" / "agent.yaml"
        if agent_path.exists():
            break
    else:
        model, _, _, _ = load_reference_policy(checkpoint)
        return "export_replica", model(obs)

    from gymnasium.spaces import Box
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.utils.model_instantiators.torch import shared_model

    cfg = yaml.safe_load(agent_path.read_text())
    if cfg["models"]["separate"]:
        raise ValueError("Only shared policy/value checkpoints are supported")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    roles = ["policy", "value"]
    parameters = [cfg["models"][role] for role in roles]
    num_actions = state["policy"]["policy_layer.weight"].shape[0]
    model = shared_model(
        observation_space=Box(-np.inf, np.inf, (obs.shape[-1],)),
        action_space=Box(-1.0, 1.0, (num_actions,)),
        device="cpu",
        structure=[params.pop("class") for params in parameters],
        roles=roles,
        parameters=parameters,
    )
    for role in roles:
        model.init_state_dict(role)
    model.load_state_dict(state["policy"])
    model.eval()
    if "state_preprocessor" in state:
        scaler = RunningStandardScaler(
            size=obs.shape[-1], device="cpu", **(cfg["agent"].get("state_preprocessor_kwargs") or {})
        )
        scaler.load_state_dict(state["state_preprocessor"])
        obs = scaler(obs)
    inputs = {"states": obs}
    mean, log_std, _ = model.compute(inputs, "policy")
    value, _ = model.compute(inputs, "value")
    return "skrl", (mean, value, log_std.expand_as(mean))


def compare_models(checkpoint: Path, ts_path: Path, onnx_path: Path, obs_path: Path, num_samples: int | None) -> dict:
    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        obs = generate_inputs(obs_path, int(ort_sess.get_inputs()[0].shape[-1]), num_samples)
        reference, (ref_mean, ref_value, ref_log_std) = reference_outputs(checkpoint, obs)

        ts_module = torch.jit.load(ts_path)
        ts_mean, ts_value, ts_log_std = ts_module(obs)

    ort_outputs = ort_sess.run(None, {"obs": obs.numpy()})
    ort_mean, ort_value, ort_log_std = [torch.from_numpy(arr) for arr in ort_outputs]

    def metrics(ref, other):
        torch.testing.assert_close(other, ref, atol=1e-4, rtol=1e-5)
        diff = other - ref
        return {
            "max_abs": float(diff.abs().max()),
            "mean_abs": float(diff.abs().mean()),
            "rmse": float(torch.sqrt((diff ** 2).mean())),
        }

    return {
        "reference": reference,
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
