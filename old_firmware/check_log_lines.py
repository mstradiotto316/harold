#!/usr/bin/env python3
import os

# Path configurations
ROOT_DIR = "/home/matteo/Desktop/Harold_V5/harold"
SIMULATION_LOGS_DIR = os.path.join(ROOT_DIR, "simulation_logs/observations.log")
ACTION_LOGS_DIR = os.path.join(ROOT_DIR, "simulation_logs/actions.log")
SCALED_ACTION_LOGS_DIR = os.path.join(ROOT_DIR, "simulation_logs/processed_actions.log")

# Check if files exist
if not os.path.exists(SIMULATION_LOGS_DIR):
    print(f"Error: Observation log file not found: {SIMULATION_LOGS_DIR}")
    exit(1)
elif not os.path.exists(ACTION_LOGS_DIR):
    print(f"Error: Action log file not found: {ACTION_LOGS_DIR}")
    exit(1)
elif not os.path.exists(SCALED_ACTION_LOGS_DIR):
    print(f"Error: Scaled action log file not found: {SCALED_ACTION_LOGS_DIR}")
    exit(1)

# Count lines in individual files
with open(SIMULATION_LOGS_DIR, 'r') as f:
    obs_lines = len(f.readlines())
    print(f"Observations log file has {obs_lines} lines")

with open(ACTION_LOGS_DIR, 'r') as f:
    action_lines = len(f.readlines())
    print(f"Actions log file has {action_lines} lines")

with open(SCALED_ACTION_LOGS_DIR, 'r') as f:
    scaled_action_lines = len(f.readlines())
    print(f"Processed actions log file has {scaled_action_lines} lines")

# Count lines when using zip
line_count = 0
with open(SIMULATION_LOGS_DIR, 'r') as obs_log_file, \
     open(ACTION_LOGS_DIR, 'r') as action_log_file, \
     open(SCALED_ACTION_LOGS_DIR, 'r') as scaled_action_log_file:
    
    for obs_line, action_line, scaled_action_line in zip(obs_log_file, action_log_file, scaled_action_log_file):
        line_count += 1
        # Print just the first and last few lines
        if line_count <= 3 or line_count >= (min(obs_lines, action_lines, scaled_action_lines) - 3):
            print(f"Line {line_count}: lengths - obs={len(obs_line)}, action={len(action_line)}, scaled={len(scaled_action_line)}")

print(f"Using zip(), processed {line_count} lines together")