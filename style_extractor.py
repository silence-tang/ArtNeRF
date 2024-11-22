"""Extract the style latent code for each image in AAHQ dataset"""
import torch
from models import encoder
import datasets
import numpy as np
import time

# 1. 节约1.5G左右的显存, 2. 缩短单个batch的推理时间(0.05s)

if __name__ == "__main__":

    device = 'cuda:1'
    # 加载StyleEncoder的预训练参数, 由于这个玩意，导致执行generator_ddp()需要9+2.x显存，溢出了
    style_encoder = encoder.StyleEncoder().to(device)
    model_dict = torch.load('ckpt/style_encoder.pt')['e']
    style_encoder.load_state_dict(model_dict)
    # 节约显存, requires_grad = False比with torch.no_grad()快0.5s/step
    for p in style_encoder.parameters():
        p.requires_grad = False

    dataset = datasets.styleFace('data/aahq', img_size=256)   # 23567
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False, pin_memory=False, num_workers=4)
    
    style_codes = []
    tic = time.time()
    for i, x_b in enumerate(dataloader):
        x_b = x_b.to(device)        # [8, 3, 256, 256]
        z_b = style_encoder(x_b)    # [8, 512]
        style_codes.append(z_b)
        print(i)
    toc = time.time()
    print("time elapsed: {:.4f}s, avg process time: {:.4f}".format(toc - tic, (toc - tic) / len(style_codes)))

    # 保存tensor
    style_codes = torch.cat(style_codes, axis=0)  # [23567, 512]
    np.savetxt("style_codes1.csv", style_codes.cpu().numpy(), delimiter=",")

    # 读取tensor
    style_codes = np.loadtxt("style_codes1.csv", delimiter = ",")
    style_codes = torch.tensor(style_codes, dtype=torch.float32).to(device)
    print(style_codes[0][0], style_codes[0][0].dtype)
