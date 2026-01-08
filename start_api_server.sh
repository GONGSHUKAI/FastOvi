#!/bin/bash
# Start FastOvi API Server

# Set environment variables
export CONFIG_PATH="${CONFIG_PATH:-configs/ovi_smallcfg.yaml}"
export CHECKPOINT_PATH="/cpfs01/gongshukai/step_distillation/logs/checkpoint_model_009000/model.pt"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure CUDA is properly installed."
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: FFmpeg not found. Please install it:"
    echo "  sudo apt update && sudo apt install ffmpeg"
    exit 1
fi

# Check if required files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "Starting FastOvi API Server..."
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo ""

# Start the server
python api_server.py
