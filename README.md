# Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning

This repository will contain the official implementation of the paper **"Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning"** presented at CoRL 2024 by Adam Fishman et al. from the University of Washington and NVIDIA Inc.

## Overview

Avoid Everything introduces a novel approach to generating collision-free motion for robotic manipulators in cluttered, partially observed environments. The system combines:

- **Motion Policy Transformer (MπFormer)**: A transformer architecture for joint space control using point clouds.
- **Refining on Optimized Policy Experts (ROPE)**: A fine-tuning procedure that refines motion policies using optimization-based demonstrations.

Key results show that Avoid Everything achieves a success rate of over 91% in challenging manipulation scenarios, significantly outperforming previous methods.

## Code Release

We are currently preparing the codebase for public release, aiming to make it available by the end of November. For updates, please follow this issue: [Code Release Issue](https://github.com/fishbotics/avoid-everything/issues/1).

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{fishman2024avoideverything,
  title={Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning},
  author={Fishman, Adam and Walsman, Aaron and Bhardwaj, Mohak and Yuan, Wentao and Sundaralingam, Balakumar and Boots, Byron and Fox, Dieter},
  booktitle={Proceedings of the Conference on Robot Learning (CoRL)},
  year={2024}
}

```
