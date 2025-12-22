#!/home/matteo/Desktop/env_isaaclab/bin/python
"""
Benchmark script to test training throughput with different num_envs values.

Usage:
    python scripts/benchmark_envs.py

This script runs short training sessions (50 iterations each) with different
environment counts and reports throughput (it/s) for each configuration.

Run this when no other training is active to get accurate measurements.
"""

import subprocess
import sys
import time
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / 'harold_isaac_lab' / 'scripts' / 'skrl' / 'train.py'
ENV_PATH = Path.home() / 'Desktop' / 'env_isaaclab' / 'bin' / 'activate'

# Test configurations
ENV_COUNTS = [2048, 4096, 6144, 8192]
ITERATIONS = 50  # Short run for benchmarking
WARMUP_ITERS = 10  # Iterations to skip for warmup


def run_benchmark(num_envs: int) -> dict:
    """Run a short training session and measure throughput."""
    print(f"\n{'='*60}")
    print(f"Testing num_envs = {num_envs}")
    print(f"{'='*60}")

    cmd = [
        'python', str(TRAIN_SCRIPT),
        '--task=Template-Harold-Direct-flat-terrain-v0',
        f'--num_envs={num_envs}',
        f'--max_iterations={ITERATIONS}',
        '--headless',
        '--rendering_mode', 'performance',
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=PROJECT_ROOT,
        )

        elapsed = time.time() - start_time
        output = result.stdout + result.stderr

        # Parse iteration speeds from output
        # Look for patterns like "20.5it/s" or "6.45it/s"
        speeds = re.findall(r'(\d+\.?\d*)\s*it/s', output)
        speeds = [float(s) for s in speeds if float(s) > 0.5]  # Filter outliers

        if len(speeds) > WARMUP_ITERS:
            # Skip warmup iterations
            speeds = speeds[WARMUP_ITERS:]
            avg_speed = sum(speeds) / len(speeds)
            max_speed = max(speeds)
            min_speed = min(speeds)
        else:
            avg_speed = max_speed = min_speed = 0.0

        # Calculate samples per second
        # Each iteration = rollouts(24) * num_envs samples
        samples_per_iter = 24 * num_envs
        samples_per_sec = avg_speed * samples_per_iter

        # Check for CUDA OOM
        oom = 'CUDA out of memory' in output or 'OutOfMemoryError' in output

        return {
            'num_envs': num_envs,
            'success': result.returncode == 0 and not oom,
            'oom': oom,
            'elapsed_sec': elapsed,
            'avg_its': avg_speed,
            'max_its': max_speed,
            'min_its': min_speed,
            'samples_per_sec': samples_per_sec,
            'data_points': len(speeds),
        }

    except subprocess.TimeoutExpired:
        return {
            'num_envs': num_envs,
            'success': False,
            'oom': False,
            'error': 'Timeout',
            'elapsed_sec': 600,
        }
    except Exception as e:
        return {
            'num_envs': num_envs,
            'success': False,
            'oom': False,
            'error': str(e),
        }


def main():
    print("Harold Environment Count Benchmark")
    print("=" * 60)
    print(f"Testing: {ENV_COUNTS}")
    print(f"Iterations per test: {ITERATIONS}")
    print(f"Warmup iterations: {WARMUP_ITERS}")
    print()

    # Check no training is running
    pid_file = Path('/tmp/harold_train.pid')
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        check = subprocess.run(['ps', '-p', pid], capture_output=True)
        if check.returncode == 0:
            print("ERROR: Training is currently running. Wait for it to finish.")
            print(f"  PID: {pid}")
            print("  Run: harold status")
            sys.exit(1)

    results = []
    for num_envs in ENV_COUNTS:
        result = run_benchmark(num_envs)
        results.append(result)

        if result.get('oom'):
            print(f"\n  CUDA OOM at {num_envs} envs - stopping benchmark")
            break

    # Print summary
    print("\n")
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print()
    print(f"{'Envs':>8} | {'it/s':>8} | {'Samples/s':>12} | {'Status':>10}")
    print("-" * 50)

    best_samples = 0
    best_envs = 0

    for r in results:
        if r['success']:
            status = "OK"
            print(f"{r['num_envs']:>8} | {r['avg_its']:>8.2f} | {r['samples_per_sec']:>12,.0f} | {status:>10}")
            if r['samples_per_sec'] > best_samples:
                best_samples = r['samples_per_sec']
                best_envs = r['num_envs']
        elif r.get('oom'):
            print(f"{r['num_envs']:>8} | {'---':>8} | {'---':>12} | {'OOM':>10}")
        else:
            print(f"{r['num_envs']:>8} | {'---':>8} | {'---':>12} | {'FAILED':>10}")

    print("-" * 50)
    print()

    if best_envs:
        print(f"RECOMMENDATION: Use num_envs = {best_envs}")
        print(f"  Peak throughput: {best_samples:,.0f} samples/sec")

        # Calculate speedup vs 4096
        baseline = next((r for r in results if r['num_envs'] == 4096 and r['success']), None)
        if baseline and best_envs != 4096:
            speedup = best_samples / baseline['samples_per_sec']
            print(f"  Speedup vs 4096: {speedup:.2f}x")

    print()
    print("To apply: Edit harold.py line 460 with recommended num_envs")


if __name__ == '__main__':
    main()
