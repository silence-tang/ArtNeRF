import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"

import argparse
import math
from torchvision.utils import save_image
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import cv2
import imageio
import datasets
from torch_ema import ExponentialMovingAverage
import time

def ten_to_cv(img):
    img = img.squeeze() * 0.5 + 0.5
    PIL_img = Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    cv2_img = cv2.cvtColor(np.asarray(PIL_img), cv2.COLOR_RGB2BGR)
    return cv2_img

def frames_to_gif(frame_list, out_video_path):
    gif = imageio.mimsave(out_video_path, frame_list, 'GIF', duration=opt.duration)
    return

def z_b_sampler(batch_size, img_size):
    dataset = datasets.AAHQ('data/aahq', 'style_codes.csv', img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=4)
    for x_b, z_b in dataloader:
        fixed_x_b = x_b
        fixed_z_b = z_b
        break   
    return fixed_x_b, fixed_z_b

def l2_penalty(param_list, l2_alpha):
    l2_loss = 0
    for param in param_list:
        l2_loss += (param ** 2).sum() / 2.0
    return l2_alpha * l2_loss


if __name__ == "__main__":
    with torch.cuda.device(2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 由于优化时需要反向传播, 导致显存消耗基本和正常训练一样多
        parser = argparse.ArgumentParser()
        parser.add_argument('--gen_path', type=str, default='experiments/artnerf_models/generator.pth')
        parser.add_argument('--output_path', type=str, default='nerf_inversion5')
        parser.add_argument('--n_iterations', type=int, default=1000)
        parser.add_argument('--img_size', type=int, default=128)
        parser.add_argument('--num_frames', type=int, default=64)
        parser.add_argument('--num_steps', type=int, default=10)
        parser.add_argument('--max_batch_size', type=int, default=2400000)
        parser.add_argument('--duration', type=float, default=0.04)
        parser.add_argument('--trajectory', type=str, default='yaw_only')

        opt = parser.parse_args()

        options = {
            'img_size': opt.img_size,
            'fov': 12,
            'ray_start': 0.88,
            'ray_end': 1.12,
            'num_steps': opt.num_steps,
            'h_stddev': 0,
            'v_stddev': 0,
            'h_mean': torch.tensor(math.pi/2).to(device),
            'v_mean': torch.tensor(math.pi/2).to(device),
            'hierarchical_sample': True,
            'sample_dist': None,
            'clamp_mode': 'relu',
            'nerf_noise': 0,
        }

        render_options = {
            'img_size': opt.img_size,
            'fov': 12,
            'ray_start': 0.88,
            'ray_end': 1.12,
            'num_steps': opt.num_steps,
            'h_stddev': 0,
            'v_stddev': 0,
            'v_mean': math.pi/2,
            'hierarchical_sample': True,
            'sample_dist': None,
            'clamp_mode': 'relu',
            'nerf_noise': 0,
            'last_back': True,
        }

        # load generator
        generator = torch.load(opt.gen_path, map_location='cpu')
        generator.to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema.load_state_dict(torch.load(opt.gen_path.split('generator')[0] + "ema.pth", map_location='cpu'))
        ema.to(device)
        ema.copy_to(generator.parameters())
        generator.set_device(device)
        generator.eval()

        # load ground truth image then save it 
        gt_img = Image.open(opt.output_path + '/gt.jpg').convert('RGB')
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        gt_img = transform(gt_img).reshape(1, 3, opt.img_size, opt.img_size).to(device)
        os.makedirs(opt.output_path, exist_ok=True)
        save_image(gt_img, opt.output_path + '/' + 'gt_img.jpg', normalize=True)

        # calculate freq and phase shifts for truncation
        z_a = torch.randn((10000, 512), device=device)
        with torch.no_grad():
            frequencies, phase_shifts = generator.siren.content_mapping_network(z_a) # [10000, 9, 256]
        w_frequencies = frequencies.mean(0, keepdim=True)          # [1, 9, 256]
        w_phase_shifts = phase_shifts.mean(0, keepdim=True)        # [1, 9, 256]

        # init freq_offsets and phase_shifts_offsets for optimization
        # 预测偏移量而非直接预测, 优化效率更高
        w_frequency_offsets = torch.zeros_like(w_frequencies).requires_grad_()      # [1, 9, 256]
        w_phase_shift_offsets = torch.zeros_like(w_phase_shifts).requires_grad_()   # [1, 9, 256]

        # init optimizer and scheduler
        optimizer = torch.optim.Adam([w_frequency_offsets, w_phase_shift_offsets], lr=0.01, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.75)
        n_iterations = opt.n_iterations

        frames = []
        tic = time.time()
        # start optimization
        for i in range(n_iterations):
            # add noise to improve robustness
            noise_w_frequencies = 0.03 * torch.randn_like(w_frequencies) * (n_iterations - i) / n_iterations
            noise_w_phase_shifts = 0.03 * torch.randn_like(w_phase_shifts) * (n_iterations - i) / n_iterations
            # 难道不需要requires_grad=False?
            gen_img, _ = generator.forward_with_frequencies(9, w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, None, None, **options)
            loss = torch.nn.MSELoss()(gen_img, gt_img).mean()
            if i % 200 == 0:
                print(i, loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 2 == 0:
                cv2_gen_img = ten_to_cv(gen_img)
                rgb_gen_img = cv2.cvtColor(cv2_gen_img, cv2.COLOR_BGR2RGB)
                frames.append(rgb_gen_img)

            # if i % 100 == 0:
            #     save_image(gen_img, f"nerf_inversion/gen_img_{i}.jpg", normalize=True)
                # with torch.no_grad():
                #     for angle in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]:
                #         img, _ = generator.staged_forward_with_frequencies(9, w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, None, None, h_mean=(math.pi/2 + angle), max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
                #         save_image(img, f"nerf_inversion/gen_img_{i}_{angle}.jpg", normalize=True)
        
        toc = time.time()
        print(toc - tic)
        # frames_to_gif(frames, opt.output_path + '/reconstructed.gif')
        # ----------------------------------------------------------------------------------------- #
        # 构造轨迹
        trajectory = []
        if opt.trajectory == 'yaw_only':
            list1 = np.linspace(-0.3, 0.3, opt.num_frames).tolist()
            list2 = np.linspace(0.3, -0.3, opt.num_frames).tolist()
            for t in list1 + list2:
                pitch = math.pi/2
                yaw = math.pi/2 + t
                trajectory.append((pitch, yaw))
                
        out_video_path_a = opt.output_path + '/fake_a_' + str(opt.img_size) + '_' + opt.trajectory + '.gif'
        out_video_path_b1 = opt.output_path + '/fake_b1_' + str(opt.img_size) + '_' + opt.trajectory + '.gif'
        out_video_path_b2 = opt.output_path + '/fake_b2_' + str(opt.img_size) + '_' + opt.trajectory + '.gif'
        out_video_path_b3 = opt.output_path + '/fake_b3_' + str(opt.img_size) + '_' + opt.trajectory + '.gif'
        frames_a = []
        frames_b1 = []
        frames_b2 = []
        frames_b3 = []

        x_b, z_b = z_b_sampler(1, opt.img_size)
        x_b = x_b.reshape(1, 3, opt.img_size, opt.img_size).to(device)
        z_b = z_b.reshape(1, 512).to(device)
        raw_style_frequencies, raw_style_phase_shifts = generator.siren.style_mapping_network(z_b)
        save_image(x_b, os.path.join(opt.output_path, f"style_img2.png"), normalize=True)
        
        # with torch.no_grad():
        #     for pitch, yaw in tqdm(trajectory):
        #         render_options['h_mean'] = yaw
        #         render_options['v_mean'] = pitch
        #         frame_a, depth_map_a = generator.staged_forward_with_frequencies(9, w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, None, None, max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
        #         cv2_img_a = ten_to_cv(frame_a)
        #         rgb_img_a = cv2.cvtColor(cv2_img_a, cv2.COLOR_BGR2RGB)
        #         frames_a.append(rgb_img_a)
        #     frames_to_gif(frames_a, out_video_path_a)

        # with torch.no_grad():
        #     for pitch, yaw in tqdm(trajectory):
        #         render_options['h_mean'] = yaw
        #         render_options['v_mean'] = pitch
        #         frame_b1, depth_map_b1 = generator.staged_forward_with_frequencies(0, w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, raw_style_frequencies, raw_style_phase_shifts, max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
        #         cv2_img_b1 = ten_to_cv(frame_b1)
        #         rgb_img_b1 = cv2.cvtColor(cv2_img_b1, cv2.COLOR_BGR2RGB)
        #         frames_b1.append(rgb_img_b1)
        #     frames_to_gif(frames_b1, out_video_path_b1)

        # with torch.no_grad():
        #     for pitch, yaw in tqdm(trajectory):
        #         render_options['h_mean'] = yaw
        #         render_options['v_mean'] = pitch
        #         frame_b2, depth_map_b2 = generator.staged_forward_with_frequencies(5, w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, raw_style_frequencies, raw_style_phase_shifts, max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
        #         cv2_img_b2 = ten_to_cv(frame_b2) 
        #         rgb_img_b2 = cv2.cvtColor(cv2_img_b2, cv2.COLOR_BGR2RGB)
        #         frames_b2.append(rgb_img_b2)
        #     frames_to_gif(frames_b2, out_video_path_b2)

        # with torch.no_grad():
        #     for pitch, yaw in tqdm(trajectory):
        #         render_options['h_mean'] = yaw
        #         render_options['v_mean'] = pitch
        #         frame_b3, depth_map_b3 = generator.staged_forward_with_frequencies(0, w_frequencies + w_frequency_offsets, w_phase_shifts + w_phase_shift_offsets, raw_style_frequencies, raw_style_phase_shifts, max_batch_size=opt.max_batch_size, lock_view_dependence=True, **render_options)
        #         cv2_img_b3 = ten_to_cv(frame_b3) 
        #         rgb_img_b3 = cv2.cvtColor(cv2_img_b3, cv2.COLOR_BGR2RGB)
        #         frames_b3.append(rgb_img_b3)
        #     frames_to_gif(frames_b3, out_video_path_b3)
