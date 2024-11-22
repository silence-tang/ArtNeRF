import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"

import argparse
import math
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import imageio
import cv2
import curriculums
import datasets
from torchvision.utils import save_image
from torch_ema import ExponentialMovingAverage

class FrequencyInterpolator1:
    def __init__(self, generator, z_a_1, z_a_2, z_b, psi=0.5):
        avg_frequencies, avg_phase_shifts = generator.generate_avg_frequencies()
        raw_content_frequencies1, raw_content_phase_shifts1 = generator.siren.content_mapping_network(z_a_1)
        self.truncated_content_frequencies1 = avg_frequencies + psi * (raw_content_frequencies1 - avg_frequencies)
        self.truncated_content_phase_shifts1 = avg_phase_shifts + psi * (raw_content_phase_shifts1 - avg_phase_shifts)
        raw_content_frequencies2, raw_content_phase_shifts2 = generator.siren.content_mapping_network(z_a_2)
        self.truncated_content_frequencies2 = avg_frequencies + psi * (raw_content_frequencies2 - avg_frequencies)
        self.truncated_content_phase_shifts2 = avg_phase_shifts + psi * (raw_content_phase_shifts2 - avg_phase_shifts)
        self.raw_style_frequencies, self.raw_style_phase_shifts = generator.siren.style_mapping_network(z_b)

    def forward(self, traj, t):
        style_frequencies = self.raw_style_frequencies
        style_phase_shifts = self.raw_style_phase_shifts
        if traj != "yaw_only":
            content_frequencies = self.truncated_content_frequencies1 * (1 - t) + self.truncated_content_frequencies2 * t
            content_phase_shifts = self.truncated_content_phase_shifts1 * (1 - t) + self.truncated_content_phase_shifts2 * t
        else:
            content_frequencies = self.truncated_content_frequencies1 * (0.5 - t) + self.truncated_content_frequencies2 * (0.5 + t)
            content_phase_shifts = self.truncated_content_phase_shifts1 * (0.5 - t) + self.truncated_content_phase_shifts2 * (0.5 + t)
        return content_frequencies, content_phase_shifts, style_frequencies, style_phase_shifts


class FrequencyInterpolator2:
    def __init__(self, generator, z_a, z_b_1, z_b_2, psi=0.5):
        avg_frequencies, avg_phase_shifts = generator.generate_avg_frequencies()
        raw_content_frequencies, raw_content_phase_shifts = generator.siren.content_mapping_network(z_a)
        self.truncated_content_frequencies = avg_frequencies + psi * (raw_content_frequencies - avg_frequencies)
        self.truncated_content_phase_shifts = avg_phase_shifts + psi * (raw_content_phase_shifts - avg_phase_shifts)
        self.raw_style_frequencies1, self.raw_style_phase_shifts1 = generator.siren.style_mapping_network(z_b_1)
        self.raw_style_frequencies2, self.raw_style_phase_shifts2 = generator.siren.style_mapping_network(z_b_2)
    
    def forward(self, traj, t):
        content_frequencies = self.truncated_content_frequencies 
        content_phase_shifts = self.truncated_content_phase_shifts
        if traj != "yaw_only":
            style_frequencies = self.raw_style_frequencies1 * (1 - t) + self.raw_style_frequencies2 * t
            style_phase_shifts = self.raw_style_phase_shifts1 * (1 - t) + self.raw_style_phase_shifts2 * t
        else:
            style_frequencies = self.raw_style_frequencies1 * (0.5 - t) + self.raw_style_frequencies2 * (0.5 + t)
            style_phase_shifts = self.raw_style_phase_shifts1 * (0.5 - t) + self.raw_style_phase_shifts2 * (0.5 + t)
        return content_frequencies, content_phase_shifts, style_frequencies, style_phase_shifts


def tensor_to_cv2_img(img):
    img = img.squeeze() * 0.5 + 0.5 # 反归一化
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

if __name__ == "__main__":
    with torch.cuda.device(9):
        parser = argparse.ArgumentParser()
        parser.add_argument('--curriculum', type=str, default='face2anime')
        parser.add_argument('--gen_path', type=str, default='experiments/artnerf_models/generator.pth')
        parser.add_argument('--output_dir', type=str, default='video_interpolation_128_1')
        parser.add_argument('--seeds', default=[0, 1, 2, 3, 4])
        parser.add_argument('--image_size', type=int, default=128)
        parser.add_argument('--num_steps', type=int, default=64)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--ray_step_multiplier', type=int, default=1)
        parser.add_argument('--num_frames', type=int, default=36)
        parser.add_argument('--duration', type=float, default=0.06)
        parser.add_argument('--psi', type=float, default=0.7)
        parser.add_argument('--max_batch_size', type=int, default=2400000)
        parser.add_argument('--depth_map', action='store_true')
        parser.add_argument('--lock_view_dependence', action='store_true')
        parser.add_argument('--trajectory', type=str, default='yaw_only')
        opt = parser.parse_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(opt.output_dir, exist_ok=True)

        curriculum = getattr(curriculums, opt.curriculum)
        curriculum['num_steps'] = opt.num_steps
        curriculum['img_size'] = opt.image_size
        curriculum['psi'] = opt.psi
        curriculum['v_stddev'] = 0
        curriculum['h_stddev'] = 0
        curriculum['lock_view_dependence'] = opt.lock_view_dependence
        curriculum['last_back'] = curriculum.get('eval_last_back', False)
        curriculum['nerf_noise'] = 0
        curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

        # 加载生成器
        generator = torch.load(opt.gen_path, map_location=torch.device(device))
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema.load_state_dict(torch.load(opt.gen_path.split('generator')[0] + "ema.pth", map_location=device))
        ema.copy_to(generator.parameters())
        generator.set_device(device)
        generator.eval()

        # 构造轨迹序列对应的相机位姿列表
        if opt.trajectory == 'front':
            trajectory = []
            for t in np.linspace(0, 1, opt.num_frames).tolist() + np.linspace(1, 0, opt.num_frames).tolist():
                pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
                yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                # fov = 12
                fov = 12 + np.sin(t * 2 * math.pi) * 2
                trajectory.append((t, pitch, yaw, fov))
                
        if opt.trajectory == 'orbit':
            trajectory = []
            for t in np.linspace(0, 1, opt.num_frames).tolist() + np.linspace(1, 0, opt.num_frames).tolist():
                pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/4
                yaw = t * 2 * math.pi
                fov = curriculum['fov']
                trajectory.append((t, pitch, yaw, fov))

        if opt.trajectory == 'yaw_only':
            trajectory = []
            for t in np.linspace(0, 1, opt.num_frames).tolist() + np.linspace(1, 0, opt.num_frames).tolist():
                pitch = math.pi/2
                yaw = curriculum['h_mean'] + (t - 0.5)
                fov = curriculum['fov']
                trajectory.append((t, pitch, yaw, fov))


        x_b, z_b = z_b_sampler(len(opt.seeds), 256)
        x_b = x_b.to(device)
        z_b = z_b.to(device)

        # 开始渲染
        for i, seed in enumerate(opt.seeds):
            print('Processing {}th video...\n'.format(i))
            frames_a = []
            frames_b1 = []
            frames_b2 = []
            out_video_path_a = opt.output_dir + '/fake_a_' + str(curriculum['img_size']) + '_' + opt.trajectory + '_{}.gif'.format(i)
            out_video_path_b1 = opt.output_dir + '/fake_b1_' + str(curriculum['img_size']) + '_' + opt.trajectory + '_{}.gif'.format(i)
            out_video_path_b2 = opt.output_dir + '/fake_b2_' + str(curriculum['img_size']) + '_' + opt.trajectory + '_{}.gif'.format(i)

            # 构造z_a和z_b
            x_b_current = x_b[i].reshape(1, 3, 256, 256)
            z_b_current = z_b[i].reshape(1, 512)
            _, z_b_next = z_b_sampler(1, 256)
            z_b_next = z_b_next.reshape(1, 512).to(device)

            z_a_current = torch.randn(1, 512, device=device)
            z_a_next = torch.randn(1, 512, device=device)
            
            # 存一下固定的B域图像
            save_image(x_b_current, os.path.join(opt.output_dir, f"style_img_{i}.png"), normalize=True)
            
            # 构造插值器
            
            # frequencyInterpolator = FrequencyInterpolator1(generator, z_a_current, z_a_next, z_b_current, psi=opt.psi)
            frequencyInterpolator = FrequencyInterpolator2(generator, z_a_current, z_b_current, z_b_next, psi=opt.psi)

            with torch.no_grad():
                for t, pitch, yaw, fov in tqdm(trajectory):
                    curriculum['h_mean'] = yaw
                    curriculum['v_mean'] = pitch
                    curriculum['fov'] = fov
                    curriculum['h_stddev'] = 0
                    curriculum['v_stddev'] = 0
                    
                    # generate fake_a
                    frame_a, _ = generator.staged_forward_with_frequencies(11, *frequencyInterpolator.forward(opt.trajectory, t), max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                    cv2_img_a = tensor_to_cv2_img(frame_a)                    # BGR
                    rgb_img_a = cv2.cvtColor(cv2_img_a, cv2.COLOR_BGR2RGB)    # RGB
                    frames_a.append(rgb_img_a)

                    # # generate fake_b1
                    # frame_b1, _ = generator.staged_forward_with_frequencies(0, *frequencyInterpolator.forward(opt.trajectory, t), max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                    # cv2_img_b1 = tensor_to_cv2_img(frame_b1)                  # BGR
                    # rgb_img_b1 = cv2.cvtColor(cv2_img_b1, cv2.COLOR_BGR2RGB)  # RGB
                    # frames_b1.append(rgb_img_b1)

                    # generate fake_b2
                    frame_b2, _ = generator.staged_forward_with_frequencies(0, *frequencyInterpolator.forward(opt.trajectory, t), max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                    cv2_img_b2 = tensor_to_cv2_img(frame_b2)                  # BGR
                    rgb_img_b2 = cv2.cvtColor(cv2_img_b2, cv2.COLOR_BGR2RGB)  # RGB
                    frames_b2.append(rgb_img_b2)

                frames_to_gif(frames_a, out_video_path_a)
                # frames_to_gif(frames_b1, out_video_path_b1)
                frames_to_gif(frames_b2, out_video_path_b2)
