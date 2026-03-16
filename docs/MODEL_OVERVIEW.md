# Model Overview

## What is GEM?

GEM (Generalist Estimation of Human Motion) is a commercial-grade monocular video 3D human pose estimation model developed by NVIDIA. It recovers full-body 77-joint motion (body, hands, and face) from monocular video using the SOMA parametric body model.

## GEM vs GENMO

GEM is the commercial successor to the research project GENMO. Key differences:

| | GENMO (Research) | GEM (This Repo) |
|---|---|---|
| Body model | SMPL (24 joints) | SOMA (77 joints: body + hands + face) |
| Modalities | Video + text + audio + music | Video only |
| License | Research only | Apache 2.0 (commercial) |
| 2D Pose Model | COCO-17 keypoints | SOMA-77 keypoints |
| Training Data | Public + internal | NVIDIA-owned only |

## SOMA Body Model

GEM uses the **SOMA** parametric body model:

- **77 joints** covering full body, hands, and face
- **MHR identity model** for body shape representation
- Bundled as a submodule in `third_party/soma`

## Architecture

| Property | Value |
|---|---|
| Parameters | ~520M |
| Architecture | 16-layer Transformer encoder |
| Positional encoding | RoPE (Rotary Position Embedding) |
| Latent dimension | 1024 |
| Attention heads | 8 |
| Decoder | Regression-based denoising decoder |
| Optimizer | AdamW (lr=2e-4) |
| Precision | 16-bit mixed |

The model takes as input video features (from SAM-3D-Body), 2D keypoints, and camera intrinsics, and outputs per-frame SOMA body parameters in both camera-relative and global coordinate frames.

## Bundled 2D Pose Model

GEM includes a trained 2D pose estimation model:

- **Backbone:** DINOv3 (ViT-based)
- **Output:** 77 SOMA keypoints per frame (x, y, confidence)
- **Usage:** Automatically invoked during the demo preprocessing pipeline

The 2D pose model is accessed via `gem.utils.vitpose_extractor.VitPoseExtractor`.
