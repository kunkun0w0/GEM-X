#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "Installing gem in editable mode..."
uv pip install -e .

echo "Installing detectron2 and SAM-3D-Body runtime deps..."
uv pip install cloudpickle fvcore iopath pycocotools braceexpand roma 'setuptools<75'
uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

echo "Environment setup complete."
