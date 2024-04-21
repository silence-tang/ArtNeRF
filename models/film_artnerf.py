from __future__ import annotations
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
Tensor = torch.Tensor
pad = F.pad

def normalize_kernel2d(input: Tensor) -> Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    norm = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm[..., None, None])

def _compute_padding(kernel_size: list[int]) -> list[int]:
    """Compute padding tuple."""
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def filter2D(input: Tensor, kernel: Tensor, border_type: str = 'reflect', normalized: bool = False, padding: str = 'same') -> Tensor:
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == 'same':
        padding_shape: list[int] = _compute_padding([height, width])
        input = pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    if padding == 'same':
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out

# init func
def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

# init func
def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

# init func
def kaiming_leaky_init_new(m):
    torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

# init func
def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    input = input.permute(0, 2, 3, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :, ]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,)
    return out[:, :, ::down_y, ::down_x]

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    return out

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k

def exists(val):
    return val is not None


class Blur1(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)


class Blur2(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
    def forward(self, coordinates):
        return coordinates * self.scale_factor


'''Map latent code from z space to W+ space'''
class MappingNetwork1(nn.Module):
    def __init__(self, z_dim=512, mid_hidden_dim=512, style_dim=256):
        super().__init__()
        self.style_dim = style_dim
        self.network = nn.Sequential(nn.Linear(z_dim, mid_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(z_dim, mid_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(z_dim, mid_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(mid_hidden_dim, 9 * 2 * self.style_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        freq_phase = self.network(z).reshape(z.shape[0], 9, -1)
        frequencies = freq_phase[..., :self.style_dim]          # [N, 9, 256]
        phase_shifts = freq_phase[..., self.style_dim:]         # [N, 9, 256]
        return frequencies, phase_shifts


class MappingNetwork2(nn.Module):
    def __init__(self, z_dim=512, mid_hidden_dim=512, style_dim=256):
        super().__init__()
        self.style_dim = style_dim
        self.network = nn.Sequential(nn.Linear(z_dim, mid_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(z_dim, mid_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(z_dim, mid_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(mid_hidden_dim, 2 * self.style_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        style = self.network(z).reshape(z.shape[0], 2, -1)     # [N, 2, 256]
        return style


# Style Blending Module
class StyleBlendingModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.ones(9 + 2, 1))

    def forward(self, add_weight_index, content_freq, content_phase_shifts, style_freq, style_phase_shifts, y_content, y_style):
        # w_new充当中间变量, 会将梯度传给self.w, 但是本身不保存其grad
        w_new = self.w.clone().to(torch.float16)
        if add_weight_index > 0:
            w_new[:add_weight_index] = 1.0
            
        freq = w_new[:9] * content_freq + (1 - w_new[:9]) * style_freq  
        phase_shifts = w_new[:9] * content_phase_shifts + (1 - w_new[:9]) * style_phase_shifts
        y = w_new[-2:] * y_content + (1 - w_new[-2:]) * y_style
        return freq.to(torch.float16), phase_shifts.to(torch.float16), y.to(torch.float16)


'''Conv2DMod from lucidrains's stylegan2-pytorch'''
class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel=1, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x_y):
        x = x_y[0]
        y = x_y[1]
        b, _, h, w = x.shape
        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)
        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)
        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)
        x = x.reshape(-1, self.filters, h, w)
        return x


'''RGBBlock from lucidrains's stylegan2-pytorch'''
class RGBBlock(nn.Module):
    def __init__(self, input_channel, upsample):
        super().__init__()
        self.input_channel = input_channel
        out_filters = 3
        self.conv = Conv2DMod(input_channel, out_filters, kernel=1, demod=False)
        self.upsample = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False), Blur1()) if upsample==True else None

    def forward(self, x_y):
        x = self.conv(x_y)
        if self.upsample != None:
            x = self.upsample(x)
        return x


'''nr for artnerf'''
class NeuralRenderer1(nn.Module):
    def __init__(self, in_chan, out_chan, style_dim):
        super().__init__()

        self.to_style1_1 = nn.Linear(style_dim, in_chan)         # for conv1
        self.to_style1_2 = nn.Linear(style_dim, in_chan//2)      # for conv2
        self.to_style1_3 = nn.Linear(style_dim, in_chan//2)      # for to_rgb[0]
        self.to_style2_1 = nn.Linear(style_dim, in_chan//2)      # for conv3
        self.to_style2_2 = nn.Linear(style_dim, out_chan)        # for conv4
        self.to_style2_3 = nn.Linear(style_dim, out_chan)        # for to_rgb[1]

        self.conv1 = Conv2DMod(in_chan, in_chan//2, 1)
        self.conv2 = Conv2DMod(in_chan//2, in_chan//2, 1)
        self.conv3 = Conv2DMod(in_chan//2, out_chan, 1)
        self.conv4 = Conv2DMod(out_chan, out_chan, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        self.block1 = nn.Sequential(self.conv1, self.act)
        self.upblock1 = nn.Sequential(self.conv2, self.upsample, self.act)
        self.block2 = nn.Sequential(self.conv3, self.act)
        self.upblock2 = nn.Sequential(self.conv4, self.upsample, self.act)

        self.to_rgb = nn.ModuleList([
            RGBBlock(in_chan//2, True),
            RGBBlock(out_chan, False)
        ])

    def forward(self, x, y):
        y1, y2 = y[:, 0, :], y[:, 1, :]
        style1_1 = self.to_style1_1(y1)
        style1_2 = self.to_style1_2(y1)
        style1_3 = self.to_style1_3(y1)
        style2_1 = self.to_style2_1(y2)
        style2_2 = self.to_style2_2(y2)
        style2_3 = self.to_style2_3(y2)

        rgb = 0
        x = self.block1([x, style1_1])                 # [N, 64, 32, 32]
        x = self.upblock1([x, style1_2])               # [N, 64, 64, 64]
        rgb += self.to_rgb[0]([x, style1_3])           # [N, 3, 128, 128]
        
        x = self.block2([x, style2_1])                 # [N, 32, 64, 64]
        x = self.upblock2([x, style2_2])               # [N, 32, 128, 128]
        rgb += self.to_rgb[1]([x, style2_3])           # [N, 3, 128, 128]

        return torch.sigmoid(rgb)                      # [N, 3, 128, 128]


'''FiLMLayer'''
class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 每个FiLM层一开始都是linear层
        self.layer = nn.Linear(input_dim, hidden_dim)

    # AMP会导致在train.py中测试和在这里测试显存占用不同    
    def forward(self, x, freq, phase_shift):
        # 先过一个线性层
        x = self.layer(x)                                      # [4,64*64*12,256], 0.1875G
        freq = freq.unsqueeze(1).expand_as(x)                  # [4,64*64*12,256], 0.1875G
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)    # [4,64*64*12,256], 0.1875G
        # 任何有关x的前向计算过程会将用到的所有变量存下来, 以便bp, 因此显存消耗大
        x = torch.sin(freq * x + phase_shift)
        return x


'''FiLM_SIREN_ArtNeRF'''
class FiLMSirenArtNeRF(nn.Module):
    def __init__(self, z_dim=512, style_dim=256, f_dim=128, hidden_dim=256, device=None):
        super().__init__()
        
        self.z_dim = z_dim
        self.style_dim = style_dim
        self.device = device

        self.content_mapping_network = MappingNetwork1(z_dim=self.z_dim, mid_hidden_dim=512, style_dim=self.style_dim)
        self.style_mapping_network = MappingNetwork1(z_dim=self.z_dim, mid_hidden_dim=512, style_dim=self.style_dim)
        self.sbm = StyleBlendingModule()

        # Main Network
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        # Mid Layers for Skip Connection
        self.to_rbg = nn.ModuleList([
            nn.Linear(hidden_dim, f_dim),
            nn.Linear(hidden_dim, f_dim),
            nn.Linear(hidden_dim, f_dim),
            nn.Linear(hidden_dim, f_dim),
            nn.Linear(hidden_dim, f_dim),
            nn.Linear(hidden_dim, f_dim),
            nn.Linear(hidden_dim, f_dim),
        ])
        
        self.to_sigma = nn.ModuleList([
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
        ])
        
        # 定义最终的RGB预测层
        self.color_head = FiLMLayer(hidden_dim + 3, hidden_dim)  # 加3是因为d是3维

        # init weights
        self.network.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.to_rbg.apply(frequency_init(25))
        self.to_sigma.apply(frequency_init(25))
        self.color_head.apply(frequency_init(25))
        # Don't worry about this
        self.gridwarper = UniformBoxWarp(0.24)


    # use skip connection and use nr
    def forward(self, add_weight_index, ray_coords, z_a, z_b, ray_directions, **kwargs):
        content_frequencies, content_phase_shifts = self.content_mapping_network(z_a)
        y_content = 0
        if z_b != None:
            style_frequencies, style_phase_shifts = self.style_mapping_network(z_b)
            y_style = self.style_mapping_network2(z_b)
        else:
            style_frequencies, style_phase_shifts, y_style = 0, 0, 0
        
        return self.forward_with_frequencies_phase_shifts(add_weight_index, ray_coords, content_frequencies, content_phase_shifts, style_frequencies, style_phase_shifts, y_content, y_style, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, add_weight_index, ray_coords, content_frequencies, content_phase_shifts, style_frequencies, style_phase_shifts, y_content, y_style, ray_directions, eps=1e-5, **kwargs):
        content_freq = content_frequencies * 15 + 30
        style_freq = style_frequencies * 15 + 30
        ray_coords = self.gridwarper(ray_coords)
        x = ray_coords                         
        sigma = 0
        f = 0
        # sbm
        freq, phase_shifts, y = self.sbm(add_weight_index, content_freq, content_phase_shifts, style_freq, style_phase_shifts, y_content, y_style)

        for i, layer in enumerate(self.network):
            x = layer(x, freq[:, i, :], phase_shifts[:, i, :])
            # if i > 0:
            # if i != 7:
            sigma += self.to_sigma[i](x) 
            f += self.to_rbg[i](x)        # [8, 64*64*12, 64]
            # else:
                # sigma += self.to_sigma[-1](x)

        # 最后再过一个rgb head
        f += self.to_rbg[-1](self.color_head(torch.cat([ray_directions, x], dim=-1), freq[:, -1, :], phase_shifts[:, -1, :]))
        
        # clamp
        f = torch.sigmoid(f) * (1 + 2 * eps) - eps
        sigma = F.relu(sigma)
        
        return torch.cat([f, sigma], dim=-1), y
    
if __name__ == "__main__":
    x = torch.randn(4, 128, 32, 32)
    y = torch.randn(4, 2, 256)
    nr = NeuralRenderer1(in_chan=128, out_chan=32, style_dim=256)
    print(nr(x, y).shape)
    
    
