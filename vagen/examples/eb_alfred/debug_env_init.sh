#!/bin/bash
# ALFRED Environment Debug and Initialization Script
# This script helps set up, debug and test the ALFRED environment for headless operation

set -x  # Print commands before execution

# Ensure Python hash seed is fixed for reproducibility
export PYTHONHASHSEED=0

# Set virtual display if needed (for headless environments)
export DISPLAY=:2  # Assumes X server is running on display 2

# Setup virtual display (uncomment if needed)
# Xvfb :2 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
# XVFB_PID=$!
# sleep 2  # Give Xvfb time to start

# Install dependencies if needed (uncomment if necessary)
# pip install pyvirtualdisplay ai2thor==5.0.0

# STEP 1: Test AI2THOR Controller in isolation
echo "=================== Testing AI2THOR Controller ==================="
python -c "
import ai2thor.controller
import os
import time
from ai2thor.wsgi_server import WsgiServer
from ai2thor.platform import CloudRendering


print('DISPLAY env var:', os.environ.get('DISPLAY'))
print('Testing AI2THOR controller...')
try:
    controller = ai2thor.controller.Controller(
        server_class=WsgiServer,
        platform=CloudRendering,
        width=300, 
        height=300,
        x_display=os.environ.get('DISPLAY', ':2'),
        quality='Very Low',
        headless=True,
        server_timeout=60.0,
        server_start_timeout=60.0,
        visibilityDistance=1.0,
        renderDepthImage=False,
        renderInstanceSegmentation=False
    )
    print('Controller initialized successfully!')
    print('Testing scene loading...')
    controller.reset('FloorPlan1')
    print('Scene loaded successfully!')
    controller.stop()
    print('AI2THOR test completed successfully!')
except Exception as e:
    print(f'ERROR: {str(e)}')
    import traceback
    traceback.print_exc()
"

# STEP 2: Create the ALFRED dataset
echo "=================== Creating ALFRED Dataset ==================="
python -m vagen.env.eb_alfred.create_dataset \
    --data_dir data/alfred \
    --start_seed 0 \
    --train_ratio 0.8 \
    --n_candidate 100 \
    --force-gen \
    --resolution 300 \
    --eval_set base \
    --exp_name test_headless \
    --down_sample_ratio 0.1 \
    --max_action_per_step 1 \
    --max_action_penalty -0.1 \
    --format_reward 0.5

# If using the virtual display, kill it when done
# kill $XVFB_PID

echo "=================== Script Execution Complete ==================="