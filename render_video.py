"""Script to render a video using a trained pi-GAN model."""

import argparse
import math
import os

import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import curriculums
from torch_ema import ExponentialMovingAverage
import cv2
import imageio
import datasets
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ten_to_cv(img):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum', type=str, default='face2anime')
    parser.add_argument('--gen_path', type=str, default='experiments/artnerf_models/generator.pth')
    parser.add_argument('--output_dir', type=str, default='multiview_videos')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--image_size_sr', type=int, default=128)
    parser.add_argument('--num_steps', type=int, default=48)
    parser.add_argument('--ray_step_multiplier', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=36)
    parser.add_argument('--duration', type=float, default=0.04)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--depth_map', action='store_true')
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--trajectory', type=str, default='yaw_only')
   
    opt = parser.parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)

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


    generator = torch.load(opt.gen_path, map_location=torch.device(device))
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema.load_state_dict(torch.load(opt.gen_path.split('generator')[0] + "ema.pth", map_location=device))
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    # 构造轨迹列表
    if opt.trajectory == 'front':
        trajectory = []
        # 保证输出的帧数为num_frames
        for t in np.linspace(0, 1, opt.num_frames):
            pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
            yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
            fov = curriculum['fov'] + 5 + np.sin(t * 2 * math.pi) * 5
            trajectory.append((pitch, yaw, fov))

    if opt.trajectory == 'yaw_only':
        trajectory = []
        list1 = np.linspace(-0.5, 0.5, opt.num_frames).tolist()
        list2 = np.linspace(0.5, -0.5, opt.num_frames).tolist()
        for t in list1 + list2 + list1 + list2:
            pitch = math.pi/2
            yaw = curriculum['h_mean'] + t
            fov = curriculum['fov']
            trajectory.append((pitch, yaw, fov))

    if opt.trajectory == 'orbits':
        trajectory = []
        for t in np.linspace(0, 1, opt.num_frames):
            pitch = math.pi/4
            yaw = t * 2 * math.pi
            fov = curriculum['fov']
            trajectory.append((pitch, yaw, fov))


    x_b, z_b = z_b_sampler(10, opt.image_size)
    x_b = x_b.to(device)
    z_b = z_b.to(device)

    infer_time1 = []
    infer_time2 = []
    infer_time3 = []
    for i in range(10):
        print('Processing {}th video...\n'.format(i))
        frames_a = []
        frames_b1 = []
        frames_b2 = []
        depths = []
        output_name = f'{i}.mp4'
        # init videoWriter
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # 视频编码方式
        out_video_path_a = opt.output_dir + '/fake_a_' + str(curriculum['img_size_sr']) + '_' + opt.trajectory + '_{}.gif'.format(i)
        out_video_path_b1 = opt.output_dir + '/fake_b1_' + str(curriculum['img_size_sr']) + '_' + opt.trajectory + '_{}.gif'.format(i)
        out_video_path_b2 = opt.output_dir + '/fake_b2_' + str(curriculum['img_size_sr']) + '_' + opt.trajectory + '_{}.gif'.format(i)
        # out = cv2.VideoWriter(out_video_path, fourcc, 24.0, (curriculum['img_size'], curriculum['img_size']))
        # writer = skvideo.io.FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
        
        cur_z_a = torch.randn((1, generator.z_dim), device=generator.device)
        cur_z_b = z_b[i].reshape(1, 512)
        cur_x_b = x_b[i].reshape(1, 3, curriculum['img_size'], curriculum['img_size'])
        # 存一下固定的B域图像
        # save_image(cur_x_b, os.path.join(opt.output_dir, f"style_img_{i}.png"), normalize=True)

        # generate fake_a
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for pitch, yaw, fov in tqdm(trajectory):
                    curriculum['h_mean'] = yaw
                    curriculum['v_mean'] = pitch
                    curriculum['fov'] = fov
                    curriculum['h_stddev'] = 0
                    curriculum['v_stddev'] = 0
                    tic = time.time()
                    _, frame_a_sr, _ = generator.forward(0, cur_z_a, cur_z_b, **curriculum)
                    # _, frame_a_sr, _ = generator.staged_forward(9, cur_z_a, None, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                    toc = time.time()
                    infer_time1.append(toc-tic)
                    # break
                    cv2_img_a = ten_to_cv(frame_a_sr)                       # BGR
                    rgb_img_a = cv2.cvtColor(cv2_img_a, cv2.COLOR_BGR2RGB)  # RGB
                    frames_a.append(rgb_img_a)
                frames_to_gif(frames_a, out_video_path_a)

        # generate fake_b1
        # with torch.no_grad():
        #     for pitch, yaw, fov in tqdm(trajectory):
        #         curriculum['h_mean'] = yaw
        #         curriculum['v_mean'] = pitch
        #         curriculum['fov'] = fov
        #         curriculum['h_stddev'] = 0
        #         curriculum['v_stddev'] = 0
        #         tic = time.time()
        #         _, frame_a_sr, _ = generator.staged_forward(0, cur_z_a, cur_z_b, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
        #         toc = time.time()
        #         infer_time2.append(toc-tic)
        #         break
                # frame_b1, depth_b1 = generator.staged_forward(0, cur_z_a, cur_z_b, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                # cv2_img_b1 = ten_to_cv(frame_b1)                          # BGR
                # rgb_img_b1 = cv2.cvtColor(cv2_img_b1, cv2.COLOR_BGR2RGB)  # RGB
                # frames_b1.append(rgb_img_b1)
            # frames_to_gif(frames_b1, out_video_path_b1)

        # generate fake_b2
        # with torch.no_grad():
        #     for pitch, yaw, fov in tqdm(trajectory):
        #         curriculum['h_mean'] = yaw
        #         curriculum['v_mean'] = pitch
        #         curriculum['fov'] = fov
        #         curriculum['h_stddev'] = 0
        #         curriculum['v_stddev'] = 0
        #         tic = time.time()
        #         _, frame_a_sr, _ = generator.staged_forward(3, cur_z_a, cur_z_b, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
        #         toc = time.time()
        #         infer_time3.append(toc-tic)
        #         break
            #     frame_b2, depth_b2 = generator.staged_forward(3, cur_z_a, cur_z_b, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
            #     cv2_img_b2 = ten_to_cv(frame_b2)                          # BGR
            #     rgb_img_b2 = cv2.cvtColor(cv2_img_b2, cv2.COLOR_BGR2RGB)  # RGB
            #     frames_b2.append(rgb_img_b2)
            # frames_to_gif(frames_b2, out_video_path_b2)
        
            # for frame in frames:
                # writer.writeFrame(np.array(frame))
                # cv2.imshow(str(i), frame) 
                # out.write(frame)    # 写入新视频文件
            # writer.close()
        # 关闭窗口
        # out.release()
        # cv2.destroyAllWindows()
    print(1.0/(sum(infer_time1)/len(infer_time1)))
    # print(1.0/(sum(infer_time2)/len(infer_time2)))
    # print(1.0/(sum(infer_time3)/len(infer_time3)))
