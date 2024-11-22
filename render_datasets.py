"""Script to render a video using a trained pi-GAN model."""

import argparse
import os

import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import numpy as np
import curriculums
from torch_ema import ExponentialMovingAverage
import cv2
import datasets
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CUDA_VISIBLE_DEVICES=1 python render_datasets.py

def ten2cv(img_ten, bgr=True):
    # chw -> hwc
    img = img_ten.squeeze(0).mul_(0.5).add_(0.5).mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if bgr:
        img = img[:, :, ::-1]
    return img

def z_b_sampler(batch_size, img_size):
    dataset = datasets.AAHQ('data/aahq', 'style_codes.csv', img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
    for x_b, z_b in dataloader:
        fixed_x_b = x_b
        fixed_z_b = z_b
        break   
    return fixed_x_b, fixed_z_b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum', type=str, default='face2anime')
    parser.add_argument('--gen_path', type=str, default='experiments/artnerf_models/generator.pth')
    parser.add_argument('--output_dir', type=str, default='gen_data')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--image_size_sr', type=int, default=128)
    parser.add_argument('--num_steps', type=int, default=48)
    parser.add_argument('--ray_step_multiplier', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--depth_map', action='store_true')
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--trajectory', type=str, default='yaw_only')
   
    opt = parser.parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'images_a'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'images_b'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'results_0'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'results_3'), exist_ok=True)


    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = opt.num_steps * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['img_size_sr'] = opt.image_size_sr
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    # 设置种子, 保证每次运行该脚本生成的z_a是一样的
    # torch.manual_seed(2)

    generator = torch.load(opt.gen_path, map_location=torch.device(device))
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema.load_state_dict(torch.load(opt.gen_path.split('generator')[0] + "ema.pth", map_location=device))
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    x_b, z_b = z_b_sampler(opt.n_samples, 256)
    x_b = x_b.to(device)
    z_b = z_b.to(device)

    # 遍历前n_samples个风格图像, 对于每个风格图像, 随机采样一个z_a以生成最终结果
    for i in range(opt.n_samples):
        print('Processing {}th data pair...'.format(i))
        cur_z_a = torch.randn((1, generator.z_dim), device=generator.device)
        cur_z_b = z_b[i].reshape(1, 512)
        cur_x_b = x_b[i].reshape(1, 3, 256, 256)
        # 保存固定的B域图像
        save_image(cur_x_b, os.path.join(opt.output_dir, 'images_b', f"{i}.png"), normalize=True)

        # fake_a_sr_all = []
        with torch.no_grad():
            idx = 0
            for pitch, yaw in [(-0.20, -0.40), (-0.10, -0.20), (0, 0), (0.10, 0.20), (0.20, 0.40)]:
                curriculum['h_mean'] = math.pi/2 + yaw
                curriculum['v_mean'] = math.pi/2 + pitch
                _, fake_a_sr, _ = generator.staged_forward(11, cur_z_a, None, **curriculum)
                save_image(fake_a_sr, os.path.join(opt.output_dir, 'images_a', f"{i}_{idx}_a.png"), nrow=1, normalize=True)
                idx += 1
        #         fake_a_sr_all.append(fake_a_sr)
        # fake_a_sr_all = torch.cat(fake_a_sr_all, axis=0)
        # save_image(fake_a_sr_all, os.path.join(opt.output_dir, 'images_a', f"{i}_a.png"), nrow=5, normalize=True)

        # 生成多视角的风格化结果并保存为1x3的网格图
        # fake_b_sr_all = []
        with torch.no_grad():
            idx = 0
            for pitch, yaw in [(-0.20, -0.40), (-0.10, -0.20), (0, 0), (0.10, 0.20), (0.20, 0.40)]:
                curriculum['h_mean'] = math.pi/2 + yaw
                curriculum['v_mean'] = math.pi/2 + pitch
                _, fake_b_sr, _ = generator.staged_forward(0, cur_z_a, cur_z_b, **curriculum)
                save_image(fake_b_sr, os.path.join(opt.output_dir, 'results_0', f"{i}_{idx}_0.png"), nrow=1, normalize=True)
                idx += 1
                # fake_b_sr_all.append(fake_b_sr)
        # fake_b_sr_all = torch.cat(fake_b_sr_all, axis=0)
        # n_row是一行展示多少个图像
        # save_image(fake_b_sr_all, os.path.join(opt.output_dir, 'results_0', f"{i}_0.png"), nrow=5, normalize=True)


        # fake_b_sr_all = []
        with torch.no_grad():
            idx = 0
            for pitch, yaw in [(-0.20, -0.40), (-0.10, -0.20), (0, 0), (0.10, 0.20), (0.20, 0.40)]:
                curriculum['h_mean'] = math.pi/2 + yaw
                curriculum['v_mean'] = math.pi/2 + pitch
                _, fake_b_sr, _ = generator.staged_forward(3, cur_z_a, cur_z_b, **curriculum)
                save_image(fake_b_sr, os.path.join(opt.output_dir, 'results_3', f"{i}_{idx}_3.png"), nrow=1, normalize=True)
                idx += 1
                # fake_b_sr_all.append(fake_b_sr)
        # fake_b_sr_all = torch.cat(fake_b_sr_all, axis=0)
        # n_row是一行展示多少个图像
        # save_image(fake_b_sr_all, os.path.join(opt.output_dir, 'results_3', f"{i}_3.png"), nrow=5, normalize=True)