#!/bin/bash
# Run all evals then stop the RunPod pod.
# Usage: bash run_evals.sh configs/run.json
# Requires OPENAI_API_KEY to be set.

set -e

CONFIG=${1:-configs/run.json}

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set"
    exit 1
fi

echo "Starting evals with config: $CONFIG"
python orchestrate.py --config "$CONFIG"

echo "All evals complete. Shutting down pod in 60 seconds..."
echo "SSH in now and run 'kill %1' if you need to cancel."
sleep 60
poweroff
