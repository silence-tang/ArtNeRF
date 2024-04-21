import os
import torch
import copy
from torchvision.utils import save_image
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from tqdm import tqdm
import datasets


def transform(x, size):
    trans = transforms.Compose([transforms.Resize((size, size))])
    return trans(x)


def style_sampler(batch_size, img_size):
    dataset = datasets.AAHQ('data/aahq', 'style_codes.csv', img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=4)
    for _, z_b in dataloader:
        fixed_z_b = z_b
        break
    return fixed_z_b    # [2096, 512]


def output_real_a_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    for x_a in dataloader:
        for img in x_a:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1
        if img_counter == num_imgs:
            break
    return


def output_real_b_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    for x_b, _, in dataloader:
        for img in x_b:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1
        if img_counter == num_imgs:
            break
    return


# 抽取8000张real_a
def setup_evaluation1(dataset_name, generated_dir, dataset_path, target_size=128, num_imgs=8000):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('EvalImages', dataset_name + '_real_a_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataset = datasets.CelebA('data/celeba', target_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False, pin_memory=False, num_workers=0)    
        # 只需要生成一次就行
        print('\n\n', 'Generating real_a images...')
        output_real_a_images(dataloader, num_imgs, real_dir)
        print('Done!')
    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir


# 抽取8000张real_b
def setup_evaluation2(dataset_name, generated_dir, dataset_path2, dataset_path3, target_size=128, num_imgs=8000):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('EvalImages', dataset_name + '_real_b_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader = datasets.get_dataset(dataset_name, batch_size=4, dataset_path2=dataset_path2, dataset_path3=dataset_path3, img_size=target_size)
        print('\n', 'Generating real_b images...')
        output_real_b_images(dataloader, num_imgs, real_dir)
        print('Done!')
    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir


# 由当前generator生成2048张fake_a
def output_images1(generator_ddp, img_size, img_size_sr, input_metadata, rank, world_size, output_dir, num_imgs=2048):
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = img_size
    metadata['img_size_sr'] = img_size_sr
    metadata['batch_size'] = 4
    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1  # 无truncation, 牺牲质量换取多样性

    img_counter = rank
    # img_counter = 0
    generator_ddp.eval()
    
    if rank == 0: pbar = tqdm('\n', "Generating fake_a images...", total = num_imgs)
    # print("Generating fake_a images...")
    with torch.no_grad():
        while img_counter < num_imgs:
            z_a = torch.randn((metadata['batch_size'], generator_ddp.module.z_dim), device=generator_ddp.module.device)
            _, gen_imgs_sr, _ = generator_ddp.module.staged_forward(9, z_a, None, **metadata)
            for img in gen_imgs_sr:
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                img_counter += world_size
                # img_counter += 1
                if rank == 0: pbar.update(world_size)
    if rank == 0:
        pbar.close()
        print("Done!")


# 由当前generator生成2048张fake_b
def output_images2(generator_ddp, img_size, img_size_sr, input_metadata, rank, world_size, output_dir, num_imgs=2048):
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = img_size
    metadata['img_size_sr'] = img_size_sr
    metadata['batch_size'] = 4
    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1

    device=generator_ddp.module.device
    img_counter = rank
    generator_ddp.eval()
    img_counter = rank
    z_b_all = style_sampler(num_imgs+48, metadata['img_size'])    # [2096, 512]

    if rank == 0: pbar = tqdm("Generating fake_b images...", total = num_imgs)
    # print("Generating fake_b images...")
    with torch.no_grad():
        idx = 0
        while img_counter < num_imgs:
            z_b = z_b_all[idx: idx + metadata['batch_size']].to(device)      # [4, 512]
            idx += metadata['batch_size']
            z_a = torch.randn((metadata['batch_size'], generator_ddp.module.z_dim), device=device)
            _, gen_imgs_sr, _ = generator_ddp.module.staged_forward(0, z_a, z_b, **metadata)
            for img in gen_imgs_sr:
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                img_counter += world_size
                if rank == 0: pbar.update(world_size)
    if rank == 0:
        pbar.close()


def calculate_fid1(dataset_name, generated_dir, target_size=128):
    real_dir = os.path.join('EvalImages', dataset_name + '_real_a_images_' + str(target_size))
    fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], target_size, 'cuda', 2048)
    torch.cuda.empty_cache()
    return fid


def calculate_fid2(dataset_name, generated_dir, target_size=128):
    real_dir = os.path.join('EvalImages', dataset_name + '_real_b_images_' + str(target_size))
    fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], target_size, 'cuda', 2048)
    torch.cuda.empty_cache()
    return fid
