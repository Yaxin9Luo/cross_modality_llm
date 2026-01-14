#!/bin/bash
# Quick setup script for cross-modality-llm project
# Usage: ./setup.sh

set -e

echo "================================"
echo "Cross-Modality LLM Setup"
echo "================================"

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "Using uv for package management..."
    uv sync
else
    echo "uv not found, using pip..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Patch timm for PyTorch 2.x compatibility
echo ""
echo "Patching timm for PyTorch 2.x compatibility..."
python scripts/patch_timm.py

echo ""
echo "================================"
echo "Setup complete!"
echo ""
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo "================================"
