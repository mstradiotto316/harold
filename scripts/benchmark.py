#!/home/matteo/Desktop/env_isaaclab/bin/python
"""
Consolidated benchmark for Harold training throughput.

Three modes:
  1. Sweep (default): Test multiple env counts, find optimal throughput
  2. Sweep + video:   Same but with video recording enabled
  3. Stress test:     Extended run at one env count for stability

Usage:
    python scripts/benchmark.py                                     # throughput sweep
    python scripts/benchmark.py --video                             # sweep with video
    python scripts/benchmark.py --stress --num-envs 16384           # 30 min stress test
    python scripts/benchmark.py --stress --num-envs 16384 --video   # stress test with video
    python scripts/benchmark.py --envs 4096,8192,16384              # custom sweep
"""

import argparse
import subprocess
import sys
import time
import re
import os
import threading
from datetime import datetime
from pathlib import Path

# Unbuffered output so background runs show progress
os.environ['PYTHONUNBUFFERED'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / 'harold_isaac_lab' / 'scripts' / 'skrl' / 'train.py'
PYTHON = Path.home() / "Desktop" / "env_isaaclab" / "bin" / "python"

DEFAULT_ENV_COUNTS = [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576]
SWEEP_ITERATIONS = 100
WARMUP_SKIP = 30
COOLDOWN_SEC = 10
ROLLOUTS_PER_ITER = 24


# --- Resource monitoring ---

def get_gpu_memory():
    """Get GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def get_ram_usage_gb():
    """Get RAM usage in GB from /proc/meminfo (more precise than free)."""
    try:
        with open('/proc/meminfo') as f:
            meminfo = f.read()
        total = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1))
        available = int(re.search(r'MemAvailable:\s+(\d+)', meminfo).group(1))
        used_kb = total - available
        return round(used_kb / 1048576, 1)
    except Exception:
        return 0.0


def get_total_ram_gb():
    """Get total RAM in GB."""
    try:
        with open('/proc/meminfo') as f:
            line = f.readline()
        total_kb = int(re.search(r'(\d+)', line).group(1))
        return round(total_kb / 1048576)
    except Exception:
        return 0


def get_gpu_name():
    """Get GPU model name."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown"


def get_gpu_total_mb():
    """Get total GPU memory in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def get_cpu_name():
    """Get CPU model name."""
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':')[1].strip()
    except Exception:
        pass
    return "Unknown"


def monitor_resources(stop_event, results):
    """Background thread to monitor peak resource usage."""
    peak_gpu = 0
    peak_ram = 0.0
    gpu_samples = []
    ram_samples = []
    while not stop_event.is_set():
        gpu = get_gpu_memory()
        ram = get_ram_usage_gb()
        peak_gpu = max(peak_gpu, gpu)
        peak_ram = max(peak_ram, ram)
        gpu_samples.append(gpu)
        ram_samples.append(ram)
        time.sleep(1)
    results['peak_gpu_mb'] = peak_gpu
    results['peak_ram_gb'] = peak_ram
    results['gpu_samples'] = gpu_samples
    results['ram_samples'] = ram_samples


# --- Training-active check ---

def check_training_active():
    """Refuse to run if training is active."""
    pid_file = Path('/tmp/harold_train.pid')
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        check = subprocess.run(['ps', '-p', pid], capture_output=True)
        if check.returncode == 0:
            print("ERROR: Training is currently running. Wait for it to finish.")
            print(f"  PID: {pid}")
            sys.exit(1)


# --- Sweep mode ---

def run_sweep_single(num_envs: int, video: bool) -> dict:
    """Run a single benchmark for given num_envs."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking num_envs = {num_envs}" + (" (with video)" if video else ""))
    print(f"{'=' * 60}")

    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT),
        '--task=Template-Harold-Direct-flat-terrain-v0',
        f'--num_envs={num_envs}',
        f'--max_iterations={SWEEP_ITERATIONS}',
        '--headless',
        '--rendering_mode', 'performance',
    ]
    if video:
        cmd.extend(['--video', '--video_interval', '50', '--video_length', '25'])

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
            timeout=900,
            cwd=PROJECT_ROOT,
        )

        elapsed = time.time() - start_time
        stop_event.set()
        monitor_thread.join()

        output = result.stdout + result.stderr

        # Parse iteration speeds from tqdm
        speeds = re.findall(r'(\d+\.?\d*)\s*it/s', output)
        speeds = [float(s) for s in speeds if 0.5 < float(s) < 200]

        # Check for OOM
        oom = 'CUDA out of memory' in output or 'OutOfMemoryError' in output

        if len(speeds) > WARMUP_SKIP and not oom:
            stable_speeds = speeds[WARMUP_SKIP:]
            avg_speed = sum(stable_speeds) / len(stable_speeds)
            min_speed = min(stable_speeds)
            max_speed = max(stable_speeds)
            samples_per_sec = avg_speed * ROLLOUTS_PER_ITER * num_envs

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
            return {
                'num_envs': num_envs,
                'success': False,
                'oom': oom,
                'elapsed_sec': elapsed,
                'peak_gpu_mb': resource_results.get('peak_gpu_mb', 0),
                'peak_ram_gb': resource_results.get('peak_ram_gb', 0),
                'error': 'OOM' if oom else f'Only {len(speeds)} speed samples (need >{WARMUP_SKIP})',
            }

    except subprocess.TimeoutExpired:
        stop_event.set()
        monitor_thread.join()
        return {'num_envs': num_envs, 'success': False, 'oom': False, 'error': 'Timeout'}
    except Exception as e:
        stop_event.set()
        monitor_thread.join()
        return {'num_envs': num_envs, 'success': False, 'oom': False, 'error': str(e)}


def run_sweep(env_counts: list[int], video: bool):
    """Run sweep across multiple env counts."""
    mode_label = "THROUGHPUT SWEEP (with video)" if video else "THROUGHPUT SWEEP (no video)"
    print("=" * 60)
    print(mode_label)
    print("=" * 60)
    print(f"Env counts: {env_counts}")
    print(f"Iterations per test: {SWEEP_ITERATIONS} ({WARMUP_SKIP} warmup)")
    print()

    baseline_gpu = get_gpu_memory()
    baseline_ram = get_ram_usage_gb()
    print(f"Baseline: GPU {baseline_gpu} MB, RAM {baseline_ram} GB")

    results = []
    for num_envs in env_counts:
        result = run_sweep_single(num_envs, video)
        results.append(result)

        if result['success']:
            print(f"  -> {result['avg_its']:.1f} it/s, "
                  f"{result['samples_per_sec']/1e6:.2f}M samples/s, "
                  f"GPU {result['peak_gpu_mb']} MB, RAM {result['peak_ram_gb']} GB")
        elif result.get('oom'):
            print(f"  -> CUDA OUT OF MEMORY (peak GPU: {result.get('peak_gpu_mb', '?')} MB)")
            break
        else:
            print(f"  -> FAILED: {result.get('error', 'unknown')}")

        print(f"  Cooling down ({COOLDOWN_SEC}s)...")
        time.sleep(COOLDOWN_SEC)

    print_summary(results, video)
    return results


def print_summary(results: list[dict], video: bool):
    """Print summary table and recommendation."""
    label = " (with video)" if video else " (no video)"
    print(f"\n{'=' * 60}")
    print(f"RESULTS{label}")
    print(f"{'=' * 60}\n")
    print(f"{'Envs':>6} | {'it/s':>7} | {'Samples/s':>10} | {'GPU MB':>7} | {'RAM GB':>6} | {'Status':>8}")
    print("-" * 65)

    best_samples = 0
    best_config = None

    for r in results:
        if r['success']:
            samples_m = r['samples_per_sec'] / 1e6
            print(f"{r['num_envs']:>6} | {r['avg_its']:>7.1f} | {samples_m:>9.2f}M | "
                  f"{r['peak_gpu_mb']:>7} | {r['peak_ram_gb']:>6.1f} | {'OK':>8}")
            if r['samples_per_sec'] > best_samples:
                best_samples = r['samples_per_sec']
                best_config = r
        elif r.get('oom'):
            print(f"{r['num_envs']:>6} | {'---':>7} | {'---':>10} | "
                  f"{r.get('peak_gpu_mb', '---'):>7} | {r.get('peak_ram_gb', '---'):>6} | {'OOM':>8}")
        else:
            print(f"{r['num_envs']:>6} | {'---':>7} | {'---':>10} | "
                  f"{'---':>7} | {'---':>6} | {'FAIL':>8}")

    print("-" * 65)

    gpu_total = get_gpu_total_mb()
    ram_total = get_total_ram_gb()

    if best_config:
        print(f"\nRECOMMENDATION: num_envs = {best_config['num_envs']}")
        print(f"  Throughput: {best_config['avg_its']:.1f} it/s "
              f"({best_samples/1e6:.2f}M samples/s)")
        if gpu_total:
            print(f"  Peak GPU: {best_config['peak_gpu_mb']} MB / {gpu_total} MB "
                  f"({100*best_config['peak_gpu_mb']/gpu_total:.0f}%)")
        if ram_total:
            print(f"  Peak RAM: {best_config['peak_ram_gb']:.1f} GB / {ram_total} GB "
                  f"({100*best_config['peak_ram_gb']/ram_total:.0f}%)")

        time_100k = 100000 / best_config['avg_its']
        print(f"  Time for 100k iterations: {time_100k/60:.1f} min")

        baseline = next((r for r in results if r['num_envs'] == 1024 and r['success']), None)
        if baseline and best_config['num_envs'] != 1024:
            speedup = best_samples / baseline['samples_per_sec']
            print(f"  Speedup vs 1024: {speedup:.2f}x")
    print()


# --- Stress test mode ---

def run_stress(num_envs: int, duration_min: int, video: bool):
    """Extended stress test at a single env count.

    Uses time-based approach: sets max_iterations very high and relies on
    the timeout to end the run after the desired duration. This avoids
    inaccurate iteration estimates (especially with video overhead).
    """
    print("=" * 60)
    video_label = " + video" if video else ""
    print(f"STRESS TEST: {num_envs} envs, {duration_min} min{video_label}")
    print("=" * 60)

    # Set iterations very high — the timeout is the real limit
    max_iterations = 999999
    timeout_sec = duration_min * 60

    print(f"\nRunning for {duration_min} min (timeout-based)...")
    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT),
        '--task=Template-Harold-Direct-flat-terrain-v0',
        f'--num_envs={num_envs}',
        f'--max_iterations={max_iterations}',
        '--headless',
        '--rendering_mode', 'performance',
    ]
    if video:
        cmd.extend(['--video', '--video_interval', '500', '--video_length', '100'])

    stop_event = threading.Event()
    resource_results = {}
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, resource_results))
    monitor_thread.start()

    start_time = time.time()
    timed_out = False

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - start_time
        stop_event.set()
        monitor_thread.join()

        output = result.stdout + result.stderr
        oom = 'CUDA out of memory' in output or 'OutOfMemoryError' in output
        crashed = result.returncode != 0

    except subprocess.TimeoutExpired:
        # Timeout is the EXPECTED exit — the training ran for the full duration
        elapsed = time.time() - start_time
        timed_out = True
        output = ""
        oom = False
        crashed = False
        stop_event.set()
        monitor_thread.join()
    except Exception as e:
        elapsed = time.time() - start_time
        stop_event.set()
        monitor_thread.join()
        print(f"\n  ERROR: {e}")
        print("  VERDICT: FAIL")
        return

    speeds = re.findall(r'(\d+\.?\d*)\s*it/s', output)
    speeds = [float(s) for s in speeds if 0.5 < float(s) < 200]

    print(f"\n{'=' * 60}")
    print("STRESS TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Duration:    {elapsed/60:.1f} min")
    print(f"  Num envs:    {num_envs}")
    print(f"  Video:       {'yes' if video else 'no'}")
    if timed_out:
        print(f"  Exit:        Ran full {duration_min} min (timeout-based stop)")
    else:
        print(f"  Exit code:   {0 if not crashed else 'non-zero'}")
    print(f"  OOM:         {'YES' if oom else 'no'}")
    print(f"  Crashed:     {'YES' if crashed else 'no'}")

    if speeds:
        stable = speeds[WARMUP_SKIP:] if len(speeds) > WARMUP_SKIP else speeds
        avg_s = sum(stable) / len(stable)
        min_s = min(stable)
        max_s = max(stable)
        print(f"\n  Throughput:")
        print(f"    avg:     {avg_s:.1f} it/s ({avg_s * ROLLOUTS_PER_ITER * num_envs / 1e6:.2f}M samples/s)")
        print(f"    min:     {min_s:.1f} it/s")
        print(f"    max:     {max_s:.1f} it/s")
        print(f"    spread:  {max_s - min_s:.1f} it/s ({100*(max_s-min_s)/avg_s:.0f}% variation)")
    elif timed_out:
        print(f"\n  Throughput: (no parsed data — output lost on timeout kill)")

    gpu_samples = resource_results.get('gpu_samples', [])
    ram_samples = resource_results.get('ram_samples', [])
    print(f"\n  Resources:")
    print(f"    Peak GPU:  {resource_results.get('peak_gpu_mb', 0)} MB")
    print(f"    Peak RAM:  {resource_results.get('peak_ram_gb', 0)} GB")
    if gpu_samples:
        avg_gpu = sum(gpu_samples) / len(gpu_samples)
        print(f"    Avg GPU:   {avg_gpu:.0f} MB")
    if ram_samples:
        avg_ram = sum(ram_samples) / len(ram_samples)
        print(f"    Avg RAM:   {avg_ram:.1f} GB")

    # Check video files if video mode
    if video:
        print(f"\n  Video check:")
        log_dir = PROJECT_ROOT / 'logs' / 'skrl' / 'harold_direct'
        if log_dir.exists():
            run_dirs = sorted(log_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if run_dirs:
                video_dir = run_dirs[0] / 'videos' / 'train'
                if video_dir.exists():
                    videos = list(video_dir.glob('*.mp4'))
                    print(f"    Videos found: {len(videos)}")
                    for v in videos[:5]:
                        size = v.stat().st_size
                        print(f"      {v.name}: {size/1024:.0f} KB")
                    if not videos:
                        print("    WARNING: No video files found!")
                else:
                    print(f"    WARNING: Video dir not found: {video_dir}")

    verdict = "PASS" if not oom and not crashed else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    if oom:
        print("    REASON: CUDA out of memory")
    if crashed:
        print("    REASON: Process crashed")


# --- Report saving ---

def save_report(results_no_video: list[dict], results_video: list[dict] | None,
                stress_summary: str | None):
    """Save benchmark report to docs/autoresearch/."""
    today = datetime.now().strftime('%Y-%m-%d')
    report_path = PROJECT_ROOT / 'docs' / 'autoresearch' / f'benchmark_{today}.md'

    gpu_name = get_gpu_name()
    gpu_total = get_gpu_total_mb()
    ram_total = get_total_ram_gb()
    cpu_name = get_cpu_name()

    lines = [
        f"# Benchmark Results — {today}",
        "",
        "## Hardware",
        "",
        f"- **GPU**: {gpu_name} ({gpu_total} MB)",
        f"- **RAM**: {ram_total} GB",
        f"- **CPU**: {cpu_name}",
        "",
    ]

    def table_from_results(results, label):
        lines = [f"## {label}", ""]
        lines.append(f"| {'Envs':>6} | {'it/s':>7} | {'Samples/s':>10} | {'GPU MB':>7} | {'RAM GB':>6} | {'Status':>6} |")
        lines.append(f"|{'-'*8}|{'-'*9}|{'-'*12}|{'-'*9}|{'-'*8}|{'-'*8}|")
        for r in results:
            if r['success']:
                samples_m = r['samples_per_sec'] / 1e6
                lines.append(
                    f"| {r['num_envs']:>6} | {r['avg_its']:>7.1f} | {samples_m:>9.2f}M | "
                    f"{r['peak_gpu_mb']:>7} | {r['peak_ram_gb']:>6.1f} | {'OK':>6} |"
                )
            elif r.get('oom'):
                lines.append(
                    f"| {r['num_envs']:>6} | {'---':>7} | {'---':>10} | "
                    f"{r.get('peak_gpu_mb', '---'):>7} | {'---':>6} | {'OOM':>6} |"
                )
            else:
                lines.append(
                    f"| {r['num_envs']:>6} | {'---':>7} | {'---':>10} | "
                    f"{'---':>7} | {'---':>6} | {'FAIL':>6} |"
                )
        lines.append("")
        return lines

    lines.extend(table_from_results(results_no_video, "Throughput Sweep (no video)"))

    if results_video:
        lines.extend(table_from_results(results_video, "Throughput Sweep (with video)"))

    # Recommendation
    best = max(
        (r for r in results_no_video if r['success']),
        key=lambda r: r['samples_per_sec'],
        default=None
    )
    if best:
        lines.append("## Recommendation")
        lines.append("")
        lines.append(f"**Optimal num_envs: {best['num_envs']}**")
        lines.append(f"- Throughput: {best['avg_its']:.1f} it/s ({best['samples_per_sec']/1e6:.2f}M samples/s)")
        lines.append(f"- Peak GPU: {best['peak_gpu_mb']} MB / {gpu_total} MB ({100*best['peak_gpu_mb']/gpu_total:.0f}%)")
        lines.append(f"- Peak RAM: {best['peak_ram_gb']:.1f} GB / {ram_total} GB ({100*best['peak_ram_gb']/ram_total:.0f}%)")
        lines.append("")

    if stress_summary:
        lines.append("## Stress Test")
        lines.append("")
        lines.append(stress_summary)
        lines.append("")

    report_path.write_text('\n'.join(lines))
    print(f"Report saved to {report_path}")


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Harold training throughput and stability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/benchmark.py                                     # throughput sweep
  python scripts/benchmark.py --video                             # sweep with video
  python scripts/benchmark.py --stress --num-envs 16384           # 30 min stress test
  python scripts/benchmark.py --stress --num-envs 16384 --video   # stress test + video
  python scripts/benchmark.py --envs 4096,8192,16384              # custom sweep
""",
    )
    parser.add_argument('--video', action='store_true',
                        help='Enable video recording during benchmark')
    parser.add_argument('--stress', action='store_true',
                        help='Run extended stress test instead of sweep')
    parser.add_argument('--num-envs', type=int, default=16384,
                        help='Env count for stress test (default: 16384)')
    parser.add_argument('--duration-min', type=int, default=30,
                        help='Stress test duration in minutes (default: 30)')
    parser.add_argument('--envs', type=str, default=None,
                        help='Comma-separated env counts for custom sweep (e.g., 4096,8192,16384)')
    return parser.parse_args()


def main():
    args = parse_args()
    check_training_active()

    print(f"Python: {PYTHON}")
    print(f"GPU: {get_gpu_name()} ({get_gpu_total_mb()} MB)")
    print(f"RAM: {get_total_ram_gb()} GB")
    print(f"CPU: {get_cpu_name()}")
    print()

    if args.stress:
        run_stress(args.num_envs, args.duration_min, args.video)
    else:
        env_counts = [int(x) for x in args.envs.split(',')] if args.envs else DEFAULT_ENV_COUNTS
        run_sweep(env_counts, args.video)


if __name__ == '__main__':
    main()
