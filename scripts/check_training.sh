#!/bin/bash
# Check Harold training status - context-efficient version
# Usage: ./scripts/check_training.sh

PID_FILE="/tmp/harold_train.pid"
LOG_FILE="/tmp/harold_train.log"

if [ ! -f "$PID_FILE" ]; then
    echo "STATUS: No training started (no PID file)"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | xargs)
    LOG_SIZE=$(wc -c < "$LOG_FILE" 2>/dev/null)
    LOG_SIZE_KB=$((LOG_SIZE / 1024))

    echo "STATUS: RUNNING (PID: $PID)"
    echo "ELAPSED: $ELAPSED"
    echo "LOG SIZE: ${LOG_SIZE_KB}KB"

    # Try to extract latest progress from the log
    # tqdm writes carriage returns, so we extract the last timestep number
    LATEST=$(grep -oP '\d+(?=/100008)' "$LOG_FILE" 2>/dev/null | tail -1)
    if [ -n "$LATEST" ]; then
        PERCENT=$((LATEST * 100 / 100008))
        echo "PROGRESS: $LATEST/100008 ($PERCENT%)"
    fi
else
    echo "STATUS: COMPLETED (process exited)"

    # Find the run directory
    LATEST_RUN=$(ls -td /home/matteo/Desktop/code_projects/harold/logs/skrl/harold_direct/2025-* 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo "RUN DIR: $(basename $LATEST_RUN)"
    fi
fi
