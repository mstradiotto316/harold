#!/bin/bash

# Default Isaac Sim installation path - modify this if your installation is different
ISAAC_SIM_PATH="/home/$USER/.local/share/ov/pkg/isaac_sim-*"

# Find the actual Isaac Sim directory
ISAAC_SIM_DIR=$(ls -d $ISAAC_SIM_PATH 2>/dev/null | head -n 1)

if [ -z "$ISAAC_SIM_DIR" ]; then
    echo "Error: Isaac Sim installation not found at $ISAAC_SIM_PATH"
    echo "Please modify this script with the correct path to your Isaac Sim installation"
    exit 1
fi

# Set environment variables
export PYTHONPATH=$ISAAC_SIM_DIR/python.${ISAAC_SIM_DIR##*-}:$PYTHONPATH
export CARB_APP_PATH=$ISAAC_SIM_DIR
export EXP_PATH=$ISAAC_SIM_DIR

echo "Isaac Sim environment variables set:"
echo "PYTHONPATH: $PYTHONPATH"
echo "CARB_APP_PATH: $CARB_APP_PATH"
echo "EXP_PATH: $EXP_PATH"

# Install the project in development mode
echo "Installing project in development mode..."
python -m pip install -e harold_isaac_lab/source/harold_isaac_lab

echo "Setup complete. You can now run your Isaac Lab code." 