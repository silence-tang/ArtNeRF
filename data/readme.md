## Notes:

- Place 128x128 natural huaman faces(around 200,000) from [CelebA](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) under `data/celeba`.
 
- Place the first 23567 128x128 natural huaman faces from CelebA under `data/celeba_mini`, which can be downloaded from [here](https://drive.google.com/drive/folders/192ciiKss36qt_IH9y-Cy7IxRfctK4Tzo?usp=sharing).

- Place 23567 128x128 artistic huaman faces from [processed AAHQ](https://drive.google.com/drive/folders/1DssaC7tmV91X4Cx5f9XFuBFb5cjTw5Ar?usp=sharing) dataset under `data/aahq`.

- You can also replace CelebA with other datasets like [FFHQ](https://github.com/NVlabs/ffhq-dataset) to achieve higher visual quality. Note that if you do this, you should change configs related to resolution(`img_size` and `img_size_sr`, etc.)
