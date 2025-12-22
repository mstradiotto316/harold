#!/home/matteo/Desktop/env_isaaclab/bin/python
"""
Test running two Isaac Sim training instances in parallel.

This tests whether we can run 2x smaller experiments simultaneously
to double hypothesis throughput.
"""

import subprocess
import sys
import time
import re
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / 'harold_isaac_lab' / 'scripts' / 'skrl' / 'train.py'

ENVS_PER_INSTANCE = 2048
ITERATIONS = 50
NUM_INSTANCES = 2


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


def run_instance(instance_id, results):
    """Run a single training instance."""
    cmd = [
        'python', str(TRAIN_SCRIPT),
        '--task=Template-Harold-Direct-flat-terrain-v0',
        f'--num_envs={ENVS_PER_INSTANCE}',
        f'--max_iterations={ITERATIONS}',
        '--headless',
        '--rendering_mode', 'performance',
    ]

    print(f"[Instance {instance_id}] Starting with {ENVS_PER_INSTANCE} envs...")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=PROJECT_ROOT,
        )

        elapsed = time.time() - start
        output = result.stdout + result.stderr

        # Parse speeds
        speeds = re.findall(r'(\d+\.?\d*)\s*it/s', output)
        speeds = [float(s) for s in speeds if 0.5 < float(s) < 100]

        if len(speeds) > 20:
            avg_speed = sum(speeds[20:]) / len(speeds[20:])
        else:
            avg_speed = 0

        oom = 'CUDA out of memory' in output
        success = result.returncode == 0 and not oom and avg_speed > 0

        results[instance_id] = {
            'success': success,
            'oom': oom,
            'elapsed': elapsed,
            'avg_its': avg_speed,
            'samples_sec': avg_speed * 24 * ENVS_PER_INSTANCE if avg_speed > 0 else 0,
        }

        print(f"[Instance {instance_id}] Finished: {avg_speed:.1f} it/s, {'OOM' if oom else 'OK'}")

    except Exception as e:
        results[instance_id] = {
            'success': False,
            'error': str(e),
        }
        print(f"[Instance {instance_id}] Error: {e}")


def main():
    print("=" * 60)
    print("PARALLEL EXPERIMENT TEST")
    print("=" * 60)
    print(f"Instances: {NUM_INSTANCES}")
    print(f"Envs per instance: {ENVS_PER_INSTANCE}")
    print(f"Total envs: {NUM_INSTANCES * ENVS_PER_INSTANCE}")
    print()

    baseline_gpu = get_gpu_memory()
    print(f"Baseline GPU: {baseline_gpu} MB")

    # Start both instances in parallel
    results = {}
    threads = []

    start_time = time.time()

    for i in range(NUM_INSTANCES):
        t = threading.Thread(target=run_instance, args=(i, results))
        threads.append(t)
        t.start()
        time.sleep(5)  # Stagger starts slightly

    # Monitor GPU while running
    peak_gpu = baseline_gpu
    while any(t.is_alive() for t in threads):
        gpu = get_gpu_memory()
        peak_gpu = max(peak_gpu, gpu)
        time.sleep(2)

    # Wait for completion
    for t in threads:
        t.join()

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Peak GPU: {peak_gpu} MB / 16376 MB ({100*peak_gpu/16376:.0f}%)")
    print(f"Total time: {total_time:.1f}s")
    print()

    all_success = all(r.get('success', False) for r in results.values())

    if all_success:
        total_samples = sum(r['samples_sec'] for r in results.values())
        print(f"PARALLEL EXPERIMENT: SUCCESS")
        print(f"  Combined throughput: {total_samples/1e6:.2f}M samples/sec")
        per_instance = [f"{r['avg_its']:.1f} it/s" for r in results.values()]
        print(f"  Per-instance: {per_instance}")

        # Compare to single 8192
        single_8192 = 2.88e6  # From benchmark
        print(f"\n  vs Single 8192 envs: {100*total_samples/single_8192:.0f}%")

        if total_samples > single_8192:
            print(f"  RECOMMENDATION: Parallel is BETTER ({total_samples/single_8192:.2f}x)")
        else:
            print(f"  RECOMMENDATION: Single 8192 is better")
    else:
        print("PARALLEL EXPERIMENT: FAILED")
        for i, r in results.items():
            if r.get('oom'):
                print(f"  Instance {i}: OOM")
            elif not r.get('success'):
                print(f"  Instance {i}: {r.get('error', 'Unknown error')}")


if __name__ == '__main__':
    main()
