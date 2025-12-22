#!/home/matteo/Desktop/env_isaaclab/bin/python
"""
Comprehensive benchmark for optimal num_envs configuration.

Tests different environment counts and measures:
- Throughput (iterations/sec, samples/sec)
- GPU memory usage
- RAM usage
- Stability

Usage:
    python scripts/benchmark_comprehensive.py
"""

import subprocess
import sys
import time
import re
import os
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / 'harold_isaac_lab' / 'scripts' / 'skrl' / 'train.py'

# Test configurations
ENV_COUNTS = [2048, 4096, 6144, 8192]
ITERATIONS = 100  # Enough for stable measurement
WARMUP_SKIP = 30  # Skip first N for warmup


def get_gpu_memory():
    """Get GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return 0


def get_ram_usage():
    """Get RAM usage in GB."""
    try:
        result = subprocess.run(['free', '-g'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        mem_line = lines[1].split()
        return int(mem_line[2])  # used
    except:
        return 0


def monitor_resources(stop_event, results):
    """Background thread to monitor peak resource usage."""
    peak_gpu = 0
    peak_ram = 0
    while not stop_event.is_set():
        gpu = get_gpu_memory()
        ram = get_ram_usage()
        peak_gpu = max(peak_gpu, gpu)
        peak_ram = max(peak_ram, ram)
        time.sleep(1)
    results['peak_gpu_mb'] = peak_gpu
    results['peak_ram_gb'] = peak_ram


def run_benchmark(num_envs: int) -> dict:
    """Run a benchmark for given num_envs."""
    print(f"\n{'='*60}")
    print(f"Benchmarking num_envs = {num_envs}")
    print(f"{'='*60}")

    cmd = [
        'python', str(TRAIN_SCRIPT),
        '--task=Template-Harold-Direct-flat-terrain-v0',
        f'--num_envs={num_envs}',
        f'--max_iterations={ITERATIONS}',
        '--headless',
        '--rendering_mode', 'performance',
    ]

    # Start resource monitor
    stop_event = threading.Event()
    resource_results = {}
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, resource_results))
    monitor_thread.start()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 min timeout
            cwd=PROJECT_ROOT,
        )

        elapsed = time.time() - start_time
        stop_event.set()
        monitor_thread.join()

        output = result.stdout + result.stderr

        # Parse iteration speeds
        speeds = re.findall(r'(\d+\.?\d*)\s*it/s', output)
        speeds = [float(s) for s in speeds if 0.5 < float(s) < 100]

        # Check for OOM
        oom = 'CUDA out of memory' in output or 'OutOfMemoryError' in output

        if len(speeds) > WARMUP_SKIP and not oom:
            # Skip warmup
            stable_speeds = speeds[WARMUP_SKIP:]
            avg_speed = sum(stable_speeds) / len(stable_speeds)
            min_speed = min(stable_speeds)
            max_speed = max(stable_speeds)

            # Calculate samples/sec (rollouts=24)
            samples_per_sec = avg_speed * 24 * num_envs

            return {
                'num_envs': num_envs,
                'success': True,
                'oom': False,
                'elapsed_sec': elapsed,
                'avg_its': avg_speed,
                'min_its': min_speed,
                'max_its': max_speed,
                'samples_per_sec': samples_per_sec,
                'peak_gpu_mb': resource_results.get('peak_gpu_mb', 0),
                'peak_ram_gb': resource_results.get('peak_ram_gb', 0),
                'data_points': len(stable_speeds),
            }
        else:
            stop_event.set()
            monitor_thread.join()
            return {
                'num_envs': num_envs,
                'success': False,
                'oom': oom,
                'elapsed_sec': elapsed,
                'peak_gpu_mb': resource_results.get('peak_gpu_mb', 0),
                'peak_ram_gb': resource_results.get('peak_ram_gb', 0),
            }

    except subprocess.TimeoutExpired:
        stop_event.set()
        monitor_thread.join()
        return {
            'num_envs': num_envs,
            'success': False,
            'oom': False,
            'error': 'Timeout',
        }
    except Exception as e:
        stop_event.set()
        monitor_thread.join()
        return {
            'num_envs': num_envs,
            'success': False,
            'error': str(e),
        }


def main():
    print("=" * 60)
    print("COMPREHENSIVE ENVIRONMENT BENCHMARK")
    print("=" * 60)
    print(f"Testing: {ENV_COUNTS}")
    print(f"Iterations per test: {ITERATIONS}")
    print(f"Warmup skip: {WARMUP_SKIP}")
    print()

    # Check baseline resources
    baseline_gpu = get_gpu_memory()
    baseline_ram = get_ram_usage()
    print(f"Baseline GPU: {baseline_gpu} MB")
    print(f"Baseline RAM: {baseline_ram} GB")

    results = []
    for num_envs in ENV_COUNTS:
        result = run_benchmark(num_envs)
        results.append(result)

        # Print immediate result
        if result['success']:
            print(f"\n  Result: {result['avg_its']:.1f} it/s, {result['samples_per_sec']/1e6:.2f}M samples/s")
            print(f"  Memory: GPU {result['peak_gpu_mb']} MB, RAM {result['peak_ram_gb']} GB")
        elif result.get('oom'):
            print(f"\n  Result: CUDA OUT OF MEMORY")
            print(f"  Peak GPU: {result.get('peak_gpu_mb', 'N/A')} MB")
            break
        else:
            print(f"\n  Result: FAILED - {result.get('error', 'unknown')}")

        # Cool down between tests
        print("  Cooling down (10s)...")
        time.sleep(10)

    # Print summary
    print("\n")
    print("=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Envs':>6} | {'it/s':>7} | {'Samples/s':>10} | {'GPU MB':>7} | {'RAM GB':>6} | {'Status':>8}")
    print("-" * 65)

    best_samples = 0
    best_config = None

    for r in results:
        if r['success']:
            status = "OK"
            samples_m = r['samples_per_sec'] / 1e6
            print(f"{r['num_envs']:>6} | {r['avg_its']:>7.1f} | {samples_m:>9.2f}M | {r['peak_gpu_mb']:>7} | {r['peak_ram_gb']:>6} | {status:>8}")

            if r['samples_per_sec'] > best_samples:
                best_samples = r['samples_per_sec']
                best_config = r
        elif r.get('oom'):
            print(f"{r['num_envs']:>6} | {'---':>7} | {'---':>10} | {r.get('peak_gpu_mb', '---'):>7} | {r.get('peak_ram_gb', '---'):>6} | {'OOM':>8}")
        else:
            print(f"{r['num_envs']:>6} | {'---':>7} | {'---':>10} | {'---':>7} | {'---':>6} | {'FAIL':>8}")

    print("-" * 65)
    print()

    if best_config:
        print("RECOMMENDATION")
        print("-" * 40)
        print(f"  Optimal num_envs: {best_config['num_envs']}")
        print(f"  Throughput: {best_config['avg_its']:.1f} it/s ({best_samples/1e6:.2f}M samples/s)")
        print(f"  Peak GPU: {best_config['peak_gpu_mb']} MB / 16376 MB ({100*best_config['peak_gpu_mb']/16376:.0f}%)")
        print(f"  Peak RAM: {best_config['peak_ram_gb']} GB / 31 GB ({100*best_config['peak_ram_gb']/31:.0f}%)")

        # Calculate expected training time for 100k timesteps
        time_100k = 100000 / best_config['avg_its']
        print(f"\n  Time for 100k timesteps: {time_100k/60:.1f} min")

        # Compare to 4096 baseline
        baseline = next((r for r in results if r['num_envs'] == 4096 and r['success']), None)
        if baseline and best_config['num_envs'] != 4096:
            speedup = best_samples / baseline['samples_per_sec']
            print(f"  Speedup vs 4096: {speedup:.2f}x")

    print()


if __name__ == '__main__':
    main()
