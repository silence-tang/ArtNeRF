import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils
from torch.nn import init

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)

class AdapterBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, input):
        return self.model(input)


class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class ResidualCCBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            # 分辨率下降一半
            CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p),
            nn.LeakyReLU(0.2, inplace=True))

        self.network.apply(kaiming_leaky_init)
        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity)/math.sqrt(2)
        return y


class CCSEncoderDiscriminator(nn.Module):
    def __init__(self, **kwargs):      # 4 -> 512
        super().__init__()
        self.step = 0
        self.epoch = 0
        self.layers = nn.ModuleList(
        [
            ResidualCCBlock(32, 64),   # 32x256x256 -> 64x128x128
            ResidualCCBlock(64, 128),  # 64x128x128 -> 128x64x64
            ResidualCCBlock(128, 256), # 128x64x64  -> 256x32x32
            ResidualCCBlock(256, 400), # 256x32x32  -> 400x16x16
            ResidualCCBlock(400, 400), # 400x16x16  -> 400x8x8
            ResidualCCBlock(400, 400), # 400x8x8    -> 400x4x4
            ResidualCCBlock(400, 400), # 400x4x4    -> 400x2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(3, 32),          # channels: 3 -> 32
            AdapterBlock(3, 64),          # channels: 3 -> 64
            AdapterBlock(3, 128),         # channels: 3 -> 128
            AdapterBlock(3, 256),         # channels: 3 -> 256
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400)          # channels: 3 -> 400
        ])

        self.final_layer = nn.Conv2d(400, 1 + 2, 2)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        
    """
    input: (N, 3, img_size, img_size)
    alpha: [0,1]之间的权重值
    """
    def forward(self, input, alpha, options=None, **kwargs):
        # 求出当前分辨率对应哪个层，分辨率越高，层数越浅
        # 例如, 64对应start=2
        start = self.img_size_to_layer[input.shape[-1]]
        # [N, 128, 64, 64]
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        
        feat1 = 0
        feat2 = 0
        # 过几个conv layer，分辨率逐渐下降
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                # x: 256x32x32
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, size=int(input.shape[2]/2), mode='nearest'))
            x = layer(x) # 最后x.shape=[N, 400, 2, 2]
            if i == 0:
                feat1 = x  # [N, 256, 32, 32]
            if i == 1:
                feat2 = x  # [N, 400, 16, 16]

        x = self.final_layer(x).reshape(x.shape[0], -1) # [N, 1+2]
        prediction = x[..., 0:1]                        # [N, 1]
        # latent = x[..., 1:257]                        # [N, 256]
        position = x[..., 1:]                           # [N, 2]
        return prediction, position, feat1.mean([2, 3]), feat2.mean([2, 3])


class CCSEncoderDiscriminator_artnerf(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.step = 0
        self.epoch = 0
        self.layers = nn.ModuleList(
        [
            ResidualCCBlock(32, 64),      # 32x256x256 -> 64x128x128
            ResidualCCBlock(64, 128),     # 64x128x128 -> 128x64x64
            ResidualCCBlock(128, 256),    # 128x64x64  -> 256x32x32
            ResidualCCBlock(256, 400),    # 256x32x32  -> 400x16x16
            ResidualCCBlock(400, 400),    # 400x16x16  -> 400x8x8
            ResidualCCBlock(400, 400),    # 400x8x8    -> 400x4x4
            ResidualCCBlock(400, 400),    # 400x4x4    -> 400x2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(3, 32),          # channels: 3 -> 32
            AdapterBlock(3, 64),          # channels: 3 -> 64
            AdapterBlock(3, 128),         # channels: 3 -> 128
            AdapterBlock(3, 256),         # channels: 3 -> 256
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400)          # channels: 3 -> 400
        ])

        self.final_layer = nn.Conv2d(400, 1+2, 2)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
    """
    input: (N, 3, img_size, img_size)
    alpha: [0,1]之间的权重值
    """
    def forward(self, input, alpha, options=None, **kwargs):
        # 求出当前分辨率对应哪个层，分辨率越高，层数越浅
        # 例如, 64对应start=2
        start = self.img_size_to_layer[input.shape[-1]]
        # [N, 128, 64, 64]
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        
        # 过几个conv layer，分辨率逐渐下降
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, size=int(input.shape[2]/2), mode='nearest'))
            x = layer(x)                                 # finally, x.shape=[N, 400, 2, 2]

        x = self.final_layer(x).reshape(x.shape[0], -1)  # [N, 3]
        pred = x[..., 0:1]    # [N, 1]
        pos = x[..., 1:]      # [N, 2]
        return pred, pos


class ProjectionDiscriminator(nn.Module):
    def __init__(self, style_dim, **kwargs):
        super().__init__()
        self.step = 0
        self.epoch = 0

        self.layers = nn.ModuleList(
        [
            ResidualCCBlock(32, 64),      # 32x256x256 -> 64x128x128
            ResidualCCBlock(64, 128),     # 64x128x128 -> 128x64x64
            ResidualCCBlock(128, 256),    # 128x64x64  -> 256x32x32
            ResidualCCBlock(256, 400),    # 256x32x32  -> 400x16x16
            ResidualCCBlock(400, 400),    # 400x16x16  -> 400x8x8
            ResidualCCBlock(400, 400),    # 400x8x8    -> 400x4x4
            ResidualCCBlock(400, 512),    # 400x4x4    -> 400x2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(3, 32),          # channels: 3 -> 32
            AdapterBlock(3, 64),          # channels: 3 -> 64
            AdapterBlock(3, 128),         # channels: 3 -> 128
            AdapterBlock(3, 256),         # channels: 3 -> 256
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400),         # channels: 3 -> 400
            AdapterBlock(3, 400)          # channels: 3 -> 400
        ])

        self.l_y = nn.Linear(style_dim, 512)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        self.final_layer = nn.Linear(512, 1)
        
    """
    input: (N, 3, img_size, img_size)
    alpha: [0,1]之间的权重值
    """
    def forward(self, input, style_latent, alpha, **kwargs):
        # 求出当前分辨率对应哪个层，分辨率越高，层数越浅
        # 例如, 64对应start=2
        start = self.img_size_to_layer[input.shape[-1]]
        # [N, 128, 64, 64]
        x = self.fromRGB[start](input)
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        
        # 过几个conv layer，分辨率逐渐下降
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                # x: [N, 256, 32, 32]
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, size=int(input.shape[2]/2), mode='nearest'))
            x = layer(x)              # 最后x.shape=[N, 512, 2, 2]

        x = torch.sum(x, dim=(2, 3))  # [N, 512]
        out = self.final_layer(x)     # [N, 1]
        out += torch.sum(self.l_y(style_latent) * x, dim=1, keepdim=True)   # [N,1]
        return out


if __name__ == "__main__":
    x = torch.randn(4, 3, 64, 64)
    y = torch.randn(4, 512)
    proj_D = ProjectionDiscriminator(style_dim=512)
    o = proj_D(x, y, alpha=0.5)
    print(o.shape)

