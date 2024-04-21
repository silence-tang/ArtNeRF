"""Implicit generator for 3D volumes"""

import random
import torch.nn as nn
import torch
import time
import curriculums
from torch.cuda.amp import autocast

from .volumetric_rendering import *

class ImplicitGenerator3d(nn.Module):
    def __init__(self, film, neural_renderer, img_size, z_dim, style_dim, f_dim, hidden_dim, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.z_dim = z_dim               # 512
        self.style_dim = style_dim       # 256
        self.f_dim = f_dim               # 64/128
        self.hidden_dim = hidden_dim     # 256
        self.film = film(z_dim=self.z_dim, style_dim=self.style_dim, f_dim=self.f_dim, hidden_dim=self.hidden_dim, device=None)
        self.neural_renderer = neural_renderer(in_chan=self.f_dim, out_chan=self.f_dim//4, style_dim=self.style_dim)
        # self.neural_renderer = neural_renderer(n_feat=self.f_dim, input_dim=self.f_dim, out_dim=3, final_actvn=True, min_feat=32, img_size=self.img_size)

    def set_device(self, device):
        self.device = device
        self.film.device = device
        self.generate_avg_frequencies()

    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""
        z_a = torch.randn((10000, self.z_dim), device=self.film.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.film.content_mapping_network1(z_a)
            y_content = self.film.content_mapping_network2(z_a)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        self.avg_y_content = y_content.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts, self.avg_y_content

    # def generate_avg_frequencies(self):
    #     """Calculates average frequencies and phase shifts"""
    #     z_a = torch.randn((10000, self.z_dim), device=self.film.device)
    #     with torch.no_grad():
    #         frequencies, phase_shifts = self.film.content_mapping_network(z_a)
    #     self.avg_frequencies = frequencies.mean(0, keepdim=True)
    #     self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
    #     return self.avg_frequencies, self.avg_phase_shifts
    
    # for neural rendering
    def forward(self, add_weight_index, z_a, z_b, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """
        batch_size = z_a.shape[0]  # 拿到batch_size
        # 生成相机光线和采样点
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            # 相机坐标系到世界坐标系的映射，涉及到相机位姿的随机采样以及采样点的perturb
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, randomize=kwargs['nerf_random'])
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # 用film模块预测各粗糙采样点的f\sigma
        coarse_output, y = self.film(add_weight_index, transformed_points, z_a, z_b, ray_directions=transformed_ray_directions_expanded)
        coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.f_dim+1)

        # 沿着光线进行重采样, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration2(coarse_output, self.f_dim, z_vals, clamp_mode=kwargs['clamp_mode'], noise_std=0.5)
                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                # Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # end new importance sampling

            # 用film模块预测各精细采样点的RGBsigma
            fine_output, y = self.film(add_weight_index, fine_points, z_a, z_b, ray_directions=transformed_ray_directions_expanded)
            fine_output = fine_output.reshape(batch_size, img_size * img_size, -1, self.f_dim+1)

            # 将粗糙和精细采样点的fsigma拼接
            # all_outputs = fine_output
            # all_z_vals = fine_z_vals
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.f_dim+1))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        # Create images with NeRF，用volumn rendereing渲染出最终图像各点的f值
        pixels, depth, weights = fancy_integration2(all_outputs, self.f_dim, all_z_vals, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=0.5)
        pixels = pixels.reshape((batch_size, img_size, img_size, self.f_dim))
        # [N, f_dim, 32, 32]
        pixels = pixels.permute(0, 3, 1, 2).contiguous()
        # neural rendering: [N, f_dim, 32, 32] -> [N, 3, 128, 128]
        pixels_sr = self.neural_renderer(pixels, y)
        
        # return pixels, torch.cat([pitch, yaw], -1)
        return pixels[:, :3, :, :] * 2 - 1, pixels_sr * 2 - 1, torch.cat([pitch, yaw], -1)  

    # def forward(self, add_weight_index, z_a, z_b, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
    #     """
    #     Generates images from a noise vector, rendering parameters, and camera distribution.
    #     Uses the hierarchical sampling scheme described in NeRF.
    #     """
    #     batch_size = z_a.shape[0]  # 拿到batch_size
    #     # 生成相机光线和采样点
    #     with torch.no_grad():
    #         points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
    #         # 相机坐标系到世界坐标系的映射，涉及到相机位姿的随机采样以及采样点的perturb
    #         transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, randomize=kwargs['nerf_random'])
    #         transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
    #         transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
    #         transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
    #         transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

    #         if lock_view_dependence:
    #             transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
    #             transformed_ray_directions_expanded[..., -1] = -1

    #     # 用film模块预测各粗糙采样点的RGB\sigma
    #     coarse_output = self.film(add_weight_index, transformed_points, z_a, z_b, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)

    #     # 沿着光线进行重采样, as described in NeRF
    #     if hierarchical_sample:
    #         with torch.no_grad():
    #             transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
    #             _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=0.5)

    #             weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

    #             # Start new importance sampling
    #             z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
    #             z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
    #             z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
    #             fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
    #             fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
    #             fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
    #             fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
    #             if lock_view_dependence:
    #                 transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
    #                 transformed_ray_directions_expanded[..., -1] = -1
    #             # end new importance sampling

    #         # 用film模块预测各精细采样点的RGBsigma
    #         fine_output = self.film(add_weight_index, fine_points, z_a, z_b, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4) # 4表示RGBsigma

    #         # 将粗糙和精细采样点的RBGsigma拼接
    #         all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
    #         all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
    #         _, indices = torch.sort(all_z_vals, dim=-2)
    #         all_z_vals = torch.gather(all_z_vals, -2, indices)
    #         all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
    #     else:
    #         all_outputs = coarse_output
    #         all_z_vals = z_vals

    #     # Create images with NeRF，用volumn rendereing渲染出最终图像各点的RGB像素值
    #     pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=0.5)
    #     pixels = pixels.reshape((batch_size, img_size, img_size, 3))
    #     pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

    #     return pixels, torch.cat([pitch, yaw], -1)

    def staged_forward(self, add_weight_index, z_a, z_b, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=70000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """
        batch_size = z_a.shape[0]
        self.generate_avg_frequencies()

        with torch.no_grad():
            raw_content_frequencies, raw_content_phase_shifts = self.film.content_mapping_network1(z_a)
            raw_y_content = self.film.content_mapping_network2(z_a)
            if z_b != None:
                raw_style_frequencies, raw_style_phase_shifts = self.film.style_mapping_network1(z_b)
                raw_y_style = self.film.style_mapping_network2(z_b)
            else:
                raw_style_frequencies, raw_style_phase_shifts, raw_y_style = 0, 0, 0
                
            # 平滑一下, 防止随机生成的z_a生成的真是人脸过于偏离数据域
            truncated_content_frequencies = self.avg_frequencies + psi * (raw_content_frequencies - self.avg_frequencies)
            truncated_content_phase_shifts = self.avg_phase_shifts + psi * (raw_content_phase_shifts - self.avg_phase_shifts)
            truncated_y_content = self.avg_y_content + psi * (raw_y_content - self.avg_y_content)

            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, randomize=kwargs['nerf_random'])
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # Sequentially evaluate film with max_batch_size to avoid OOM
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], self.f_dim+1), device=self.device)
            # 需要根据mapping_network2的输出shape作出相应的更改
            y = torch.zeros((batch_size, 2, 256), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    if z_b != None:
                        coarse_output[b:b+1, head:tail], y[b:b+1, ...] = self.film.forward_with_frequencies_phase_shifts(add_weight_index, transformed_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], raw_style_frequencies[b:b+1], raw_style_phase_shifts[b:b+1], truncated_y_content[b:b+1], raw_y_style[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    else:
                        coarse_output[b:b+1, head:tail], y[b:b+1, ...] = self.film.forward_with_frequencies_phase_shifts(add_weight_index, transformed_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], 0, 0, truncated_y_content[b:b+1], 0, ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.f_dim+1) 
            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration2(coarse_output, self.f_dim, z_vals, clamp_mode=kwargs['clamp_mode'], noise_std=0.5)
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    # Start new importance sampling
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach().to(self.device)
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    # end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                # Sequentially evaluate film with max_batch_size to avoid OOM
                fine_output = torch.zeros((batch_size, fine_points.shape[1], self.f_dim+1), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        if z_b != None:
                            fine_output[b:b+1, head:tail], _ = self.film.forward_with_frequencies_phase_shifts(add_weight_index, fine_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], raw_style_frequencies[b:b+1], raw_style_phase_shifts[b:b+1], truncated_y_content[b:b+1], raw_y_style[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        else:
                            fine_output[b:b+1, head:tail], _ = self.film.forward_with_frequencies_phase_shifts(add_weight_index, fine_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], 0, 0, truncated_y_content[b:b+1], 0, ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, self.f_dim+1)
                
                # all_outputs = fine_output
                # all_z_vals = fine_z_vals
                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.f_dim+1))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals

            pixels, depth, weights = fancy_integration2(all_outputs, self.f_dim, all_z_vals, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=0.5)
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            pixels = pixels.reshape((batch_size, img_size, img_size, self.f_dim))
            # [N, f_dim, 16, 16]
            pixels = pixels.permute(0, 3, 1, 2).contiguous()
            # neural rendering
            pixels_sr = self.neural_renderer(pixels, y).to(torch.float32)

        return (pixels[:, :3, :, :] * 2 - 1).cpu(), (pixels_sr * 2 - 1).cpu(), depth_map

    # def staged_forward(self, add_weight_index, z_a, z_b, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=70000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
    #     """
    #     Similar to forward but used for inference.
    #     Calls the model sequencially using max_batch_size to limit memory usage.
    #     """
    #     batch_size = z_a.shape[0]
    #     self.generate_avg_frequencies()

    #     with torch.no_grad():
    #         raw_content_frequencies, raw_content_phase_shifts = self.film.content_mapping_network(z_a)
    #         if z_b != None:
    #             raw_style_frequencies, raw_style_phase_shifts = self.film.style_mapping_network(z_b)
    #         else:
    #             raw_style_frequencies, raw_style_phase_shifts = 0, 0
    #         # 平滑一下, 防止随机生成的z_a生成的真是人脸过于偏离数据域
    #         truncated_content_frequencies = self.avg_frequencies + psi * (raw_content_frequencies - self.avg_frequencies)
    #         truncated_content_phase_shifts = self.avg_phase_shifts + psi * (raw_content_phase_shifts - self.avg_phase_shifts)

    #         points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
    #         transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, randomize=kwargs['nerf_random'])
    #         transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
    #         transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
    #         transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
    #         transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

    #         if lock_view_dependence:
    #             transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
    #             transformed_ray_directions_expanded[..., -1] = -1

    #         # Sequentially evaluate film with max_batch_size to avoid OOM
    #         coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
    #         for b in range(batch_size):
    #             head = 0
    #             while head < transformed_points.shape[1]:
    #                 tail = head + max_batch_size
    #                 if z_b != None:
    #                     coarse_output[b:b+1, head:tail] = self.film.forward_with_frequencies_phase_shifts(add_weight_index, transformed_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], raw_style_frequencies[b:b+1], raw_style_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
    #                 else:
    #                     coarse_output[b:b+1, head:tail] = self.film.forward_with_frequencies_phase_shifts(add_weight_index, transformed_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], 0, 0, ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
    #                 head += max_batch_size

    #         coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)

    #         if hierarchical_sample:
    #             with torch.no_grad():
    #                 transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
    #                 _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=0.5)
    #                 weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
    #                 # Start new importance sampling
    #                 z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
    #                 z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
    #                 z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
    #                 fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
    #                                  num_steps, det=False).detach().to(self.device)
    #                 fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
    #                 fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
    #                 fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
    #                 # end new importance sampling

    #             if lock_view_dependence:
    #                 transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
    #                 transformed_ray_directions_expanded[..., -1] = -1

    #             # Sequentially evaluate film with max_batch_size to avoid OOM
    #             fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
    #             for b in range(batch_size):
    #                 head = 0
    #                 while head < fine_points.shape[1]:
    #                     tail = head + max_batch_size
    #                     if z_b != None:
    #                         fine_output[b:b+1, head:tail] = self.film.forward_with_frequencies_phase_shifts(add_weight_index, fine_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], raw_style_frequencies[b:b+1], raw_style_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
    #                     else:
    #                         fine_output[b:b+1, head:tail] = self.film.forward_with_frequencies_phase_shifts(add_weight_index, fine_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], 0, 0, ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
    #                     head += max_batch_size

    #             fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
    #             all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
    #             all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
    #             _, indices = torch.sort(all_z_vals, dim=-2)
    #             all_z_vals = torch.gather(all_z_vals, -2, indices)
    #             all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
    #         else:
    #             all_outputs = coarse_output
    #             all_z_vals = z_vals

    #         pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=0.5)
    #         depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
    #         pixels = pixels.reshape((batch_size, img_size, img_size, 3))
    #         pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

    #     return pixels, depth_map

    def staged_forward_with_frequencies(self, add_weight_index, truncated_content_frequencies, truncated_content_phase_shifts, truncated_y_content, raw_style_frequencies, raw_style_phase_shifts, raw_y_style, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.7, lock_view_dependence=False, max_batch_size=70000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        batch_size = truncated_content_frequencies.shape[0]
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, randomize=False)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # coarse sampling
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], self.f_dim+1), device=self.device)
            y = torch.zeros((batch_size, 2, 256), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    if raw_style_frequencies != None:
                        coarse_output[b:b+1, head:tail], y[b:b+1, ...] = self.film.forward_with_frequencies_phase_shifts(add_weight_index, transformed_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], raw_style_frequencies[b:b+1], raw_style_phase_shifts[b:b+1], truncated_y_content[b:b+1], raw_y_style[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    else:
                        coarse_output[b:b+1, head:tail], y[b:b+1, ...] = self.film.forward_with_frequencies_phase_shifts(add_weight_index, transformed_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], 0, 0, truncated_y_content[b:b+1], 0, ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.f_dim+1) 

            # coarse sampling
            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration2(coarse_output, self.f_dim, z_vals, clamp_mode=kwargs['clamp_mode'], noise_std=0.5)
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    # Start new importance sampling
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach().to(self.device)
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    # end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                fine_output = torch.zeros((batch_size, fine_points.shape[1], self.f_dim+1), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        if raw_style_frequencies != None:
                            fine_output[b:b+1, head:tail], _ = self.film.forward_with_frequencies_phase_shifts(add_weight_index, fine_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], raw_style_frequencies[b:b+1], raw_style_phase_shifts[b:b+1], truncated_y_content[b:b+1], raw_y_style[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        else:
                            fine_output[b:b+1, head:tail], _ = self.film.forward_with_frequencies_phase_shifts(add_weight_index, fine_points[b:b+1, head:tail], truncated_content_frequencies[b:b+1], truncated_content_phase_shifts[b:b+1], 0, 0, truncated_y_content[b:b+1], 0, ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, self.f_dim+1)
                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.f_dim+1))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals

            pixels, depth, weights = fancy_integration2(all_outputs, self.f_dim, all_z_vals, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=0.5)
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            pixels = pixels.reshape((batch_size, img_size, img_size, self.f_dim))
            # [N, f_dim, 16, 16]
            pixels = pixels.permute(0, 3, 1, 2).contiguous()
            # neural rendering
            pixels_sr = self.neural_renderer(pixels, y).to(torch.float32)

        return (pixels[:, :3, :, :] * 2 - 1).cpu(), (pixels_sr * 2 - 1).cpu(), depth_map


    def forward_with_frequencies(self, add_weight_index, raw_content_frequencies, raw_content_phase_shifts, raw_style_frequencies, raw_style_phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        batch_size = raw_content_frequencies.shape[0]

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, randomize=False)
        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)
        
        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1
        
        if raw_style_frequencies != None:
            coarse_output = self.film.forward_with_frequencies_phase_shifts(add_weight_index, transformed_points, raw_content_frequencies, raw_content_phase_shifts, raw_style_frequencies, raw_style_phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)
        else:
            coarse_output = self.film.forward_with_frequencies_phase_shifts(add_weight_index, transformed_points, raw_content_frequencies, raw_content_phase_shifts, 0, 0, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)
        
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=0.5)
                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                # end new importance sampling
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
            
            if raw_style_frequencies != None:
                fine_output = self.film.forward_with_frequencies_phase_shifts(add_weight_index, fine_points, raw_content_frequencies, raw_content_phase_shifts, raw_style_frequencies, raw_style_phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
            else:
                fine_output = self.film.forward_with_frequencies_phase_shifts(add_weight_index, fine_points, raw_content_frequencies, raw_content_phase_shifts, 0, 0, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=0.5)
        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        return pixels, torch.cat([pitch, yaw], -1)

if __name__ == "__main__":
    pass
