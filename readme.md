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

# Examples

## Main results
|                         Style Image                          |                            fake_a                            |                        fake_b (i = 0)                        |                        fake_b (i = 3)                        |                        fake_b (i = 11)                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![style_img_1.png](https://img1.imgtp.com/2023/10/06/elbVRreS.png) | ![fake_a_128_yaw_only_1.gif](https://img1.imgtp.com/2023/10/06/KV7pj7qo.gif) | ![fake_b1_128_yaw_only_1.gif](https://img1.imgtp.com/2023/10/06/NYooAsMS.gif) | ![fake_b2_128_yaw_only_1.gif](https://img1.imgtp.com/2023/10/06/XqzwpQcE.gif) | ![fake_a_128_yaw_only_1.gif](https://img1.imgtp.com/2023/10/06/KV7pj7qo.gif) |
| ![style_img_3.png](https://img1.imgtp.com/2023/10/06/wr3sDwvw.png) | ![fake_a_128_yaw_only_3.gif](https://img1.imgtp.com/2023/10/06/pkT1TAlv.gif) | ![fake_b1_128_yaw_only_3.gif](https://img1.imgtp.com/2023/10/06/tGKawRRK.gif) | ![fake_b2_128_yaw_only_3.gif](https://img1.imgtp.com/2023/10/06/nOuWCJ26.gif) | ![fake_a_128_yaw_only_3.gif](https://img1.imgtp.com/2023/10/06/pkT1TAlv.gif) |
| ![style_img_5.png](https://img1.imgtp.com/2023/10/06/rmi8CqeF.png) | ![fake_a_128_yaw_only_5.gif](https://img1.imgtp.com/2023/10/06/GYQxhyMi.gif) | ![fake_b1_128_yaw_only_5.gif](https://img1.imgtp.com/2023/10/06/zPodizDD.gif) | ![fake_b2_128_yaw_only_5.gif](https://img1.imgtp.com/2023/10/06/zoq5f1J4.gif) | ![fake_a_128_yaw_only_5.gif](https://img1.imgtp.com/2023/10/06/GYQxhyMi.gif) |
| ![style_img_6.png](https://img1.imgtp.com/2023/10/06/VgXvDkEY.png) | ![fake_a_128_yaw_only_6.gif](https://img1.imgtp.com/2023/10/06/FhzNeEpx.gif) | ![fake_b1_128_yaw_only_6.gif](https://img1.imgtp.com/2023/10/06/trqq6gkY.gif) | ![fake_b2_128_yaw_only_6.gif](https://img1.imgtp.com/2023/10/06/xJapC6mQ.gif) | ![fake_a_128_yaw_only_6.gif](https://img1.imgtp.com/2023/10/06/FhzNeEpx.gif) |
| ![style_img_7.png](https://img1.imgtp.com/2023/10/06/QdJzrVF0.png) | ![fake_a_128_yaw_only_7.gif](https://img1.imgtp.com/2023/10/06/T5faV2dt.gif) | ![fake_b1_128_yaw_only_7.gif](https://img1.imgtp.com/2023/10/06/UAucv8PL.gif) | ![fake_b2_128_yaw_only_7.gif](https://img1.imgtp.com/2023/10/06/8htVCjtZ.gif) | ![fake_a_128_yaw_only_7.gif](https://img1.imgtp.com/2023/10/06/T5faV2dt.gif) |

<br>

## Here shows some other stylized face avatars with different resolutions.

### 64×64

| ![64_yaw_only_0.gif](https://img1.imgtp.com/2023/10/06/tW8Axhp1.gif) | ![64_yaw_only_1.gif](https://img1.imgtp.com/2023/10/06/SmovwLVd.gif) | ![64_yaw_only_2.gif](https://img1.imgtp.com/2023/10/06/BwFpY2a2.gif) | ![64_yaw_only_3.gif](https://img1.imgtp.com/2023/10/06/foFvOtdn.gif) | ![64_yaw_only_4.gif](https://img1.imgtp.com/2023/10/06/ICUXRkVk.gif) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![64_yaw_only_5.gif](https://img1.imgtp.com/2023/10/06/VcECeDHb.gif) | ![64_yaw_only_6.gif](https://img1.imgtp.com/2023/10/06/GXXajnTI.gif) | ![64_yaw_only_7.gif](https://img1.imgtp.com/2023/10/06/GqAOEcil.gif) | ![64_yaw_only_8.gif](https://img1.imgtp.com/2023/10/06/XkSJJCbF.gif) | ![64_yaw_only_9.gif](https://img1.imgtp.com/2023/10/06/FcCQ7uO2.gif) |

<br>

### 128x128

| ![128_yaw_only_0.gif](https://img1.imgtp.com/2023/10/06/7yEFtdJP.gif) | ![128_yaw_only_1.gif](https://img1.imgtp.com/2023/10/06/I10Lpqoc.gif) | ![128_yaw_only_2.gif](https://img1.imgtp.com/2023/10/06/ADWYQ81g.gif) | ![128_yaw_only_5.gif](https://img1.imgtp.com/2023/10/06/tv3LVkAZ.gif) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![128_yaw_only_9.gif](https://img1.imgtp.com/2023/10/06/VS1VZDqL.gif) | ![128_yaw_only_6.gif](https://img1.imgtp.com/2023/10/06/KeLPQcZj.gif) | ![128_yaw_only_8.gif](https://img1.imgtp.com/2023/10/06/QLebod7W.gif) | ![128_yaw_only_3.gif](https://img1.imgtp.com/2023/10/06/zbcV9Jnl.gif) |

<br>

## Latent Space Interpolation

<br>

Following traditional face synthesis models like StyleGAN, we can perform interpolation between any two latent codes.

| ![fake_a_128_front_1.gif](https://img1.imgtp.com/2023/10/06/z5Aqa25s.gif) | ![fake_a_128_front_3.gif](https://img1.imgtp.com/2023/10/06/aRzPYIhP.gif) | ![fake_a_128_front_4.gif](https://img1.imgtp.com/2023/10/06/UHVAcvQL.gif) | ![fake_a_128_yaw_only_4.gif](https://img1.imgtp.com/2023/10/06/mfMmSxm0.gif) | ![fake_b1_128_front_3.gif](https://img1.imgtp.com/2023/10/06/Pivq6fDH.gif) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![fake_b1_128_front_1.gif](https://img1.imgtp.com/2023/10/06/eCEJtkTa.gif) | ![interp_z_b_yaw_only_6.gif](https://img1.imgtp.com/2023/10/06/QeNSM3Ni.gif) | ![interp_z_b_yaw_only_7.gif](https://img1.imgtp.com/2023/10/06/XPH233yp.gif) | ![interp_z_b_yaw_only_8.gif](https://img1.imgtp.com/2023/10/06/XmHRtL8k.gif) | ![fake_b1_128_front_2.gif](https://img1.imgtp.com/2023/10/06/FD2T7dzm.gif) |
| ![interp_z_b_yaw_only_1.gif](https://img1.imgtp.com/2023/10/06/COGHZxZD.gif) | ![interp_z_b_yaw_only_4.gif](https://img1.imgtp.com/2023/10/06/324qYBNX.gif) | ![interp_z_b_yaw_only_10.gif](https://img1.imgtp.com/2023/10/06/HNSMkWZc.gif) | ![fake_a_128_yaw_only_0.gif](https://img1.imgtp.com/2023/10/06/Kxm9Cy7N.gif) | ![fake_a_128_yaw_only_2.gif](https://img1.imgtp.com/2023/10/06/R0rGmRr6.gif)&nbsp; |


| ![fake_b1_128_front_1.gif](https://img1.imgtp.com/2023/10/06/eCEJtkTa.gif) | ![interp_z_b_yaw_only_6.gif](https://img1.imgtp.com/2023/10/06/QeNSM3Ni.gif) | ![interp_z_b_yaw_only_7.gif](https://img1.imgtp.com/2023/10/06/XPH233yp.gif) | ![interp_z_b_yaw_only_8.gif](https://img1.imgtp.com/2023/10/06/XmHRtL8k.gif) | ![fake_b1_128_front_2.gif](https://img1.imgtp.com/2023/10/06/FD2T7dzm.gif) |
| ![interp_z_b_yaw_only_1.gif](https://img1.imgtp.com/2023/10/06/COGHZxZD.gif) | ![interp_z_b_yaw_only_4.gif](https://img1.imgtp.com/2023/10/06/324qYBNX.gif) | ![interp_z_b_yaw_only_10.gif](https://img1.imgtp.com/2023/10/06/HNSMkWZc.gif) | ![fake_a_128_yaw_only_0.gif](https://img1.imgtp.com/2023/10/06/Kxm9Cy7N.gif) | ![fake_a_128_yaw_only_2.gif](https://img1.imgtp.com/2023/10/06/R0rGmRr6.gif)&nbsp; |

