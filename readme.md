# Environment
- GPU: 1 NVIDIA GeForce RTX 2080 Ti with 11GB memory is enough.
- OS: Linux Ubuntu 18.04 LTS
- IDE: Visual Studio Code 2022.09
- Others: Python3.7 + PyTorch1.8.1 + CUDA10.1

# Preparation
- To prepare data and pretraind models, please check all the file folders in this project and follow the guidance in `readme.md`.
- To accelerate the training process, we precompute the 512-dim style code for every artistic human face, you can download [style_codes.csv](https://drive.google.com/file/d/1Y-ZCIDe_uYC4YealGT-jw1iZw66-EZj9/view?usp=sharing) and place it under `ArtNerf/`.

# Training
-  The model is trained by conducting a two-stage training strategy: pretraining on CelebA and fine-tuning on both AAHQ and CelebA.
-  The whole model is composed of 1 generator and 3 dicriminators. `disc_real` guides the `gen` to generate natural human faces and `disc_style` guides the `gen` to generate stylized human faces. `disc_latent` helps ensure the style-consistency between generated faces and ref faces.
-  We use a style blending module to help stabilize the cross-domain transfer learning process and allow users to change the extent to which the generated images is stylized(level can be changed from 0 to 11).


