#!/bin/bash
# Harold Training Helper - Runs training in background to avoid context overflow
# Usage: ./scripts/run_experiment.sh [max_iterations] [extra_args...]

set -e

# Configuration
HAROLD_DIR="/home/matteo/Desktop/code_projects/harold"
ENV_PATH="$HOME/Desktop/env_isaaclab/bin/activate"
LOG_FILE="/tmp/harold_train.log"
PID_FILE="/tmp/harold_train.pid"

# Default iterations (100k timesteps = ~4167 iterations with rollouts=24)
MAX_ITERATIONS="${1:-4167}"
shift 2>/dev/null || true

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ERROR: Training already running (PID: $OLD_PID)"
        echo "Check status: tail -5 $LOG_FILE"
        echo "Kill it: kill $OLD_PID"
        exit 1
    fi
fi

# Clear old log
> "$LOG_FILE"

echo "Starting Harold training in background..."
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"

# Run training in background
cd "$HAROLD_DIR"
source "$ENV_PATH"

nohup python harold_isaac_lab/scripts/skrl/train.py \
    --task=Template-Harold-Direct-flat-terrain-v0 \
    --num_envs 1024 \
    --max_iterations "$MAX_ITERATIONS" \
    --headless \
    "$@" \
    > "$LOG_FILE" 2>&1 &

# Save PID
echo $! > "$PID_FILE"
echo "Training started with PID: $(cat $PID_FILE)"
echo ""
echo "Monitor progress: tail -f $LOG_FILE"
echo "Check status: ./scripts/check_training.sh"
