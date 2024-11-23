import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from collections import namedtuple
import torchvision.models as models

import sys
sys.path.append("/home/xxx/ArtNeRF-main") # 需要设置合适的路径
from vgg_base import vgg2


def fused_bias_act(inp, bias, negative_slope=0.2, scale=2 ** 0.5):
    fused_act = F.leaky_relu(inp + bias, negative_slope=negative_slope, inplace=False) * scale
    return fused_act


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return scale * F.leaky_relu(input + bias.view((1, -1)+(1,)*(len(input.shape)-2)), negative_slope=negative_slope)


FeatureOutput = namedtuple(
    "FeatureOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.vgg_layers = vgg2.vgg19_backbone().vgg19_model.features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return FeatureOutput(**output)


class StyleEmbedder(nn.Module):
    def __init__(self):
        super(StyleEmbedder, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.eval()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def forward(self, img):
        N = img.shape[0]
        features = self.feature_extractor(self.avg_pool(img))
        grams = []
        for feature in features:
            gram = gram_matrix(feature)
            grams.append(gram.view(N, -1))
        out = torch.cat(grams, dim=1)
        return out


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=512, n_mlp=4):
        super().__init__()
        self.style_dim = style_dim
        e_dim = 610304
        # 得到Gram矩阵concat的610304维长向量
        self.embedder = StyleEmbedder()
        # 4层MLP把610304维向量映射到512维
        layers = []
        layers.append(EqualLinear(e_dim, style_dim, lr_mul=1, activation='fused_lrelu'))
        for _ in range(n_mlp - 2):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=1, activation='fused_lrelu'))
        layers.append(EqualLinear(style_dim, style_dim, lr_mul=1, activation=None))
        self.embedder_mlp = nn.Sequential(*layers)

    def forward(self, image):
        # 将风格图像编码为512维的隐变量z
        z_embed = self.embedder_mlp(self.embedder(image))  # [N, 512]
        return z_embed

if __name__ == "__main__":
    
    enc = StyleEncoder().to('cuda')
    model_dict = torch.load('ckpt/style_encoder.pt')['e']
    enc.load_state_dict(model_dict)
    x = torch.randn(4, 3, 64, 64).cuda()
    embedding = enc.embedder(x)
    print(embedding.shape)
