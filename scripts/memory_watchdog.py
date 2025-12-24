#!/usr/bin/env python3
"""
Memory watchdog for Isaac Lab training.

Monitors system memory and kills training if it exceeds safe thresholds
to prevent OOM-induced system hangs.

Usage:
    python scripts/memory_watchdog.py --pid <training_pid>
    python scripts/memory_watchdog.py --auto  # Find training process automatically
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Marker file to signal watchdog killed training
KILL_MARKER = Path("/tmp/harold_watchdog_killed.json")

# Safety thresholds - only intervene at truly dangerous levels
RAM_WARN_PERCENT = 85      # Warn at 85% RAM usage
RAM_KILL_PERCENT = 95      # Kill only when RAM critically full
SWAP_KILL_PERCENT = 70     # Kill only when swap heavily used (thrashing imminent)
CHECK_INTERVAL = 10        # Check every 10 seconds

def get_memory_info():
    """Get current memory usage."""
    with open('/proc/meminfo', 'r') as f:
        meminfo = {}
        for line in f:
            parts = line.split()
            key = parts[0].rstrip(':')
            value = int(parts[1])  # kB
            meminfo[key] = value

    total_ram = meminfo['MemTotal']
    available_ram = meminfo['MemAvailable']
    used_ram = total_ram - available_ram
    ram_percent = (used_ram / total_ram) * 100

    total_swap = meminfo.get('SwapTotal', 0)
    free_swap = meminfo.get('SwapFree', 0)
    used_swap = total_swap - free_swap
    swap_percent = (used_swap / total_swap * 100) if total_swap > 0 else 0

    return {
        'ram_used_gb': used_ram / 1024 / 1024,
        'ram_total_gb': total_ram / 1024 / 1024,
        'ram_percent': ram_percent,
        'swap_used_gb': used_swap / 1024 / 1024,
        'swap_total_gb': total_swap / 1024 / 1024,
        'swap_percent': swap_percent,
    }

def find_training_pid():
    """Find Isaac Lab training process."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'train.py.*Template-Harold'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            return int(pids[0]) if pids[0] else None
    except Exception:
        pass
    return None

def kill_process_tree(pid, reason: str, mem_info: dict):
    """Kill process and all children, leave marker for agents."""
    # Write marker file so agents know training was killed
    marker_data = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'pid': pid,
        'reason': reason,
        'ram_percent': mem_info.get('ram_percent', 0),
        'swap_percent': mem_info.get('swap_percent', 0),
        'ram_used_gb': mem_info.get('ram_used_gb', 0),
        'swap_used_gb': mem_info.get('swap_used_gb', 0),
    }
    KILL_MARKER.write_text(json.dumps(marker_data, indent=2))

    try:
        # Kill entire process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(2)
        # Force kill if still running
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Fall back to killing just the process
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass

def log(msg, level='INFO'):
    """Log with timestamp."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='Memory watchdog for training')
    parser.add_argument('--pid', type=int, help='PID to monitor')
    parser.add_argument('--auto', action='store_true', help='Auto-find training process')
    parser.add_argument('--ram-kill', type=int, default=RAM_KILL_PERCENT,
                       help=f'Kill at this RAM %% (default: {RAM_KILL_PERCENT})')
    parser.add_argument('--swap-kill', type=int, default=SWAP_KILL_PERCENT,
                       help=f'Kill at this swap %% (default: {SWAP_KILL_PERCENT})')
    args = parser.parse_args()

    if args.auto:
        pid = find_training_pid()
        if not pid:
            log("No training process found", 'ERROR')
            sys.exit(1)
        log(f"Auto-detected training PID: {pid}")
    elif args.pid:
        pid = args.pid
    else:
        log("Specify --pid or --auto", 'ERROR')
        sys.exit(1)

    log(f"Watchdog started for PID {pid}")
    log(f"Thresholds: RAM={args.ram_kill}%, Swap={args.swap_kill}%")

    warned = False

    while True:
        # Check if process still exists
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log("Training process ended normally")
            break
        except PermissionError:
            pass  # Process exists but we can't signal it

        mem = get_memory_info()

        # Check swap first (most dangerous)
        if mem['swap_percent'] > args.swap_kill:
            log(f"CRITICAL: Swap at {mem['swap_percent']:.1f}% (>{args.swap_kill}%)", 'CRITICAL')
            log(f"  RAM: {mem['ram_used_gb']:.1f}/{mem['ram_total_gb']:.1f} GB ({mem['ram_percent']:.1f}%)")
            log(f"  Swap: {mem['swap_used_gb']:.1f}/{mem['swap_total_gb']:.1f} GB")
            log("Killing training to prevent system hang...")
            kill_process_tree(pid, reason='swap_pressure', mem_info=mem)
            log("Training killed due to swap pressure")
            break

        # Check RAM
        if mem['ram_percent'] > args.ram_kill:
            log(f"CRITICAL: RAM at {mem['ram_percent']:.1f}% (>{args.ram_kill}%)", 'CRITICAL')
            log(f"  RAM: {mem['ram_used_gb']:.1f}/{mem['ram_total_gb']:.1f} GB")
            log(f"  Swap: {mem['swap_used_gb']:.1f}/{mem['swap_total_gb']:.1f} GB ({mem['swap_percent']:.1f}%)")
            log("Killing training to prevent OOM...")
            kill_process_tree(pid, reason='ram_pressure', mem_info=mem)
            log("Training killed due to RAM pressure")
            break

        # Warning level
        if mem['ram_percent'] > RAM_WARN_PERCENT and not warned:
            log(f"WARNING: RAM at {mem['ram_percent']:.1f}%", 'WARN')
            warned = True
        elif mem['ram_percent'] < RAM_WARN_PERCENT:
            warned = False

        time.sleep(CHECK_INTERVAL)

    sys.exit(0)

if __name__ == '__main__':
    main()
