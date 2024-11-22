import argparse
import os
import torch
from torchvision.utils import save_image
from torch_ema import ExponentialMovingAverage
import datasets
import curriculums
from models import encoder
import cv2
from torchvision import transforms

    
def z_b_sampler(batch_size, img_size):
    dataset = datasets.AAHQ('data/aahq', 'style_codes.csv', img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=4)
    for x_b, z_b in dataloader:
        fixed_x_b = x_b
        fixed_z_b = z_b
        break   
    return fixed_x_b, fixed_z_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum', type=str, default='face2anime')
    parser.add_argument('--gen_path', type=str, default='experiments/artnerf_models/generator.pth')
    parser.add_argument('--output_dir', type=str, default='fake_imgs')
    parser.add_argument('--ref_img_dir', type=str, default='ref_imgs')
    parser.add_argument('--lock_view_dependence', action='store_true')

    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(0)

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['img_size'] = 32
    curriculum['img_size_sr'] = 128
    curriculum['num_steps'] = 48
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    
    os.makedirs(opt.output_dir, exist_ok=True)

    # 加载模型
    generator = torch.load(opt.gen_path, map_location=torch.device(device))
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema.load_state_dict(torch.load(opt.gen_path.split('generator')[0] + "ema.pth", map_location=device))
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    # 加载style_encoder
    style_encoder = encoder.StyleEncoder().to(device)
    model_dict = torch.load('ckpt/style_encoder.pt')['e']
    style_encoder.load_state_dict(model_dict)
    # 节约显存, requires_grad = False比with torch.no_grad()快0.5s/step
    for p in style_encoder.parameters():
        p.requires_grad = False


    # 构造位姿集合
    yaw_angles = [-0.30, 0.30]
    yaw_angles = [y + curriculum['h_mean'] for y in yaw_angles]
    pitch_angles = [-0.15, 0.15]
    pitch_angles = [p + curriculum['v_mean'] for p in pitch_angles]

    # 准备固定的z_a
    fixed_z_a = torch.randn((7, 512), device=device)
    gen_fake_a_all = []
    for i in range(fixed_z_a.shape[0]):
        tmp_z_a = fixed_z_a[i].reshape(1, 512)
        _, fake_a_sr, _ = generator.staged_forward(11, tmp_z_a, None, **curriculum)
        gen_fake_a_all.append(fake_a_sr)

    gen_fake_a_all = torch.cat(gen_fake_a_all, axis=0)
    save_image(gen_fake_a_all, os.path.join(opt.output_dir, f"fake_a_sr.png"), nrow=1, normalize=True)


    curriculum['h_mean'] = yaw_angles[0]
    curriculum['v_mean'] = pitch_angles[0]
    # 生成x_b对应的z_b
    ref_img_all = os.listdir(opt.ref_img_dir)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((256, 256))])
    trans2 = transforms.Compose([transforms.Resize((128, 128))])
    gen_imgs_all = []
    with torch.no_grad():
        for i in range(len(ref_img_all)):
            tmp_ref_img = cv2.imread(os.path.join(opt.ref_img_dir, ref_img_all[i]))
            tmp_ref_img = tmp_ref_img[:,:,::-1].copy() # BGR->RGB
            tmp_ref_img = trans(tmp_ref_img).to(device).reshape(1, 3, 256, 256)
            # 每行第一个图是参考图像
            gen_imgs_all.append(trans2(tmp_ref_img))
            tmp_z_b = style_encoder(tmp_ref_img)
            for j in range(fixed_z_a.shape[0]):
                tmp_z_a = fixed_z_a[j].reshape(1, 512)
                _, fake_b_sr_0, _ = generator.staged_forward(0, tmp_z_a, tmp_z_b, **curriculum)
                gen_imgs_all.append(fake_b_sr_0.to(device))

        gen_imgs_all = torch.cat(gen_imgs_all, axis=0)
        save_image(gen_imgs_all, os.path.join(opt.output_dir, f"fake_b_sr_0_left.png"), nrow=8, normalize=True)

    # 换个姿势
    curriculum['h_mean'] = yaw_angles[1]
    curriculum['v_mean'] = pitch_angles[1]
    # 生成x_b对应的z_b
    gen_imgs_all = []
    with torch.no_grad():
        for i in range(len(ref_img_all)):
            tmp_ref_img = cv2.imread(os.path.join(opt.ref_img_dir, ref_img_all[i]))
            tmp_ref_img = tmp_ref_img[:,:,::-1].copy() # BGR->RGB
            tmp_ref_img = trans(tmp_ref_img).to(device).reshape(1, 3, 256, 256)
            # 每行第一个图是参考图像
            gen_imgs_all.append(trans2(tmp_ref_img))
            tmp_z_b = style_encoder(tmp_ref_img)
            for j in range(fixed_z_a.shape[0]):
                tmp_z_a = fixed_z_a[j].reshape(1, 512)
                _, fake_b_sr_0, _ = generator.staged_forward(0, tmp_z_a, tmp_z_b, **curriculum)
                gen_imgs_all.append(fake_b_sr_0.to(device))

        gen_imgs_all = torch.cat(gen_imgs_all, axis=0)
        save_image(gen_imgs_all, os.path.join(opt.output_dir, f"fake_b_sr_0_right.png"), nrow=8, normalize=True)

