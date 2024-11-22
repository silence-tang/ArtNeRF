# Environment
- GPU: 1 NVIDIA GeForce RTX 2080 Ti with 11GB memory is enough.
- OS: Linux Ubuntu 18.04 LTS
- IDE: Visual Studio Code 2022.09
- Others: Python3.7 + PyTorch1.8.1 + CUDA10.1

# Preparation
## Data
- Place 128x128 natural huaman faces(around 200,000) from [CelebA](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) under `data/celeba/`.
- Place the first 23567 128x128 natural huaman faces from CelebA under `data/celeba_mini`, which can be downloaded from [here](https://drive.google.com/drive/folders/192ciiKss36qt_IH9y-Cy7IxRfctK4Tzo?usp=sharing).
- Place 23567 128x128 artistic huaman faces from [processed AAHQ](https://drive.google.com/drive/folders/1DssaC7tmV91X4Cx5f9XFuBFb5cjTw5Ar?usp=sharing) dataset under `data/aahq/`.
- You can also replace CelebA with other datasets like [FFHQ](https://github.com/NVlabs/ffhq-dataset) to achieve higher visual quality. Note that if you do this, you should change configs related to resolution(`img_size` and `img_size_sr`, etc.)
## Pretrained models
- Download and unzip the [file](https://drive.google.com/file/d/1PZ-OxxAotbyD-4dnONN54sTJW9kvPxwW/view?usp=sharing) and put `style_encoder.pt` under `ckpt/`
- Download and unzip the [file](https://drive.google.com/drive/folders/1LA0Lowx3l5_nUIRwqSOqzHDS4XYkRHeR?usp=sharing). Place all the related file folders (`artnerf_models`, `base_models`) under `experiments/`
- Place `vgg19-dcbb9e9d.pth` (which can be downloaded from PyTorch official website) in this directory.
- To accelerate the training process, we precompute the 512-dim style code for every artistic human face, you can download [style_codes.csv](https://drive.google.com/file/d/1Y-ZCIDe_uYC4YealGT-jw1iZw66-EZj9/view?usp=sharing) and place it under `ArtNerf/`.

# Training
-  The model is trained by conducting a two-stage training strategy: pretraining on CelebA and fine-tuning on both AAHQ and CelebA.
-  The whole model is composed of 1 generator and 3 dicriminators. `disc_real` guides the `gen` to generate natural human faces and `disc_style` guides the `gen` to generate stylized human faces. `disc_latent` helps ensure the style-consistency between generated faces and ref faces.
-  We use a style blending module to help stabilize the cross-domain transfer learning process and allow users to change the extent to which the generated images is stylized(level can be changed from 0 to 11).


