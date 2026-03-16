# Related Projects

## GENMO

**GENMO: A GENeralist Model for Human MOtion** is the research predecessor of GEM, presented at ICCV 2025.

- **Repository:** [https://github.com/NVlabs/GENMO](https://github.com/NVlabs/GENMO)
- **Differences:** GENMO uses the SMPL body model (24 joints) and supports multi-modal conditioning (text, audio, music). GEM focuses on video-only estimation with the SOMA body model (77 joints) and is licensed for commercial use.

## SOMA

**SOMA** is the parametric body model used by GEM. It extends beyond SMPL to include hand and face articulation with 77 joints and the MHR identity model.

- Bundled as a submodule in `third_party/soma`

## Links

- **Project Page:** [https://research.nvidia.com/labs/dair/gem/](https://research.nvidia.com/labs/dair/gem/)
- **Paper:** [https://arxiv.org/abs/2505.01425](https://arxiv.org/abs/2505.01425)
