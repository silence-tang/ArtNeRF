import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import PIL
import numpy as np

class CelebA(Dataset):
    """CelelebA Dataset"""
    def __init__(self, dataset_path, img_size_sr, **kwargs):
        super().__init__()
        self.data = glob.glob(os.path.join(dataset_path) + "/*.*")
        assert len(self.data) > 0, "Can't find real faces data, please check dataset_path."
        self.transform = transforms.Compose([transforms.Resize(320), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size_sr, img_size_sr))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        return X

class AAHQ(Dataset):
    def __init__(self, dataset_path2, dataset_path3, img_size, **kwargs): 
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size))])
        self.style_face  = glob.glob(os.path.join(dataset_path2) + "/*.*")
        self.style_code = np.loadtxt(dataset_path3, delimiter = ",")
        self.style_code = torch.tensor(self.style_code, dtype=torch.float32)  # [23567, 512]
        assert len(self.style_face) > 0, "Can't find style face data, please check dataset_path."
        assert len(self.style_code) > 0, "Can't find style code data, please check dataset_path."
        
    def __getitem__(self, index):
        style_face = PIL.Image.open(self.style_face[index])
        style_face = self.transform(style_face)
        style_code = self.style_code[index]
        return style_face, style_code

    def __len__(self):
        return len(self.style_face)


class FaceStylization(Dataset):
    def __init__(self, dataset_path1, dataset_path2, dataset_path3, img_size_sr, **kwargs): 
        super().__init__()
        self.transform1 = transforms.Compose([transforms.Resize(320), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size_sr, img_size_sr))])
        self.transform2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size_sr, img_size_sr))])
        self.real_face = glob.glob(os.path.join(dataset_path1) + "/*.*")
        self.style_face  = glob.glob(os.path.join(dataset_path2) + "/*.*")
        self.style_code = np.loadtxt(dataset_path3, delimiter = ",")
        self.style_code = torch.tensor(self.style_code, dtype=torch.float32)  # [23567,512]
        assert len(self.real_face) > 0, "Can't find real face data, please check dataset_path."
        assert len(self.style_face) > 0, "Can't find style face data, please check dataset_path."
        assert len(self.style_code) > 0, "Can't find style code data, please check dataset_path."
        
    def __getitem__(self, index):
        real_face = PIL.Image.open(self.real_face[index])
        style_face = PIL.Image.open(self.style_face[index])
        style_code = self.style_code[index]
        real_face = self.transform1(real_face)
        style_face = self.transform2(style_face)
        return real_face, style_face, style_code

    def __len__(self):
        return len(self.real_face)


class StyleFace(Dataset):
    def __init__(self, dataset_path, img_size): 
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size))])
        self.styleface = glob.glob(os.path.join(dataset_path) + "/*.*")
        assert len(self.styleface) > 0, "Can't find style faces data, please check dataset_path."
        
    def __getitem__(self, index):
        styleface = PIL.Image.open(self.styleface[index])
        styleface = self.transform(styleface)
        return styleface

    def __len__(self):
        return len(self.styleface)

class RealFace(Dataset):
    def __init__(self, dataset_path, img_size, mode, **kwargs): 
        super().__init__()
        # 验证集图像无需翻转
        if mode == 'train':
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size))])
            self.realface = glob.glob(os.path.join(dataset_path, "%s" % mode, 'trainA') + "/*.*")
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size))])
            self.realface = glob.glob(os.path.join(dataset_path, "%s" % mode, 'testA') + "/*.*")
        
        assert len(self.realface) > 0, "Can't find real faces data, please check dataset_path."
        
    def __getitem__(self, index):
        face_A = PIL.Image.open(self.realface[index])
        face_A = self.transform(face_A)
        return face_A

    def __len__(self):
        return len(self.realface)

class AniFace(Dataset):
    def __init__(self, dataset_path, img_size, mode, **kwargs): 
        super().__init__()
        # 验证集图像无需翻转
        if mode == 'train':
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=0)])
            self.aniface  = glob.glob(os.path.join(dataset_path, "%s" % mode, 'trainB') + "/*.*")
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=0)])
            self.aniface  = glob.glob(os.path.join(dataset_path, "%s" % mode, 'testB') + "/*.*")

        assert len(self.aniface) > 0, "Can't find anime faces data, please check dataset_path."
        
    def __getitem__(self, index):
        face_B = PIL.Image.open(self.aniface[index])
        face_B = self.transform(face_B)
        return face_B

    def __len__(self):
        return len(self.aniface)

# 单卡
def get_dataset(name, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )
    return dataloader

def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    # 在train中执行get_dataset_distributed()函数时，已经定义了CelebA类
    # globals()[name]调用CelebA类，而后面(**kwargs)是传入metadata字典作为带变量名的参数
    dataset = globals()[name](**kwargs)
    # 分布式sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    # 构造dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,  # Sampler option is mutually exclusive with shuffle
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    return dataloader

