import torch
import torch.nn as nn
import torchvision.models as models

class vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_model = models.vgg19()
        self.pretrained_file = torch.load('pretrained/vgg19-dcbb9e9d.pth')  # 需要设置正确的路径
        self.vgg_model.load_state_dict(self.pretrained_file)
        self.module_all = torch.nn.Sequential(*list(self.vgg_model.children()))

    def show_modules(self):
        print(self.module_all[0])

    def norm(self, x):
        InstanceNorm = nn.InstanceNorm2d(num_features=x.shape[1])
        return InstanceNorm(x)

    def forward(self, x):
        return self.vgg_model(x)


if __name__ == "__main__":
    vgg = vgg19()
    x = torch.randn(2,3,64,64)
    f = vgg(x)
    print(f.shape)
    # print(vgg.show_modules())
    # state_dict存的是所有层的卷积核的权重矩阵和偏置系数，也即模型参数
    # print(vgg.module_all.state_dict().keys()) 
    # print(vgg.module_all.state_dict()['0.0.weight'][0][0], '\n', vgg.module_all.state_dict()['0.23.weight'][0][0])
