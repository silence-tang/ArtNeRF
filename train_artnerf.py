import argparse
import os
import math
from tqdm import tqdm
from datetime import datetime
import copy
import logging
import random
import copy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from generators import generators_artnerf
from discriminators import discriminators
from models import film_artnerf

import fid_evaluation
import datasets
import curriculums
from torch_ema import ExponentialMovingAverage

from torch.utils.tensorboard import SummaryWriter


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def dis_loss(fake_pred, real_pred):
    return torch.mean(torch.nn.functional.softplus(fake_pred)) + torch.mean(torch.nn.functional.softplus(-real_pred))


def gen_loss(fake_pred):
    return torch.mean(torch.nn.functional.softplus(-fake_pred))


def pos_loss(pos, pos_pred):
    return torch.nn.MSELoss()(pos, pos_pred)


def z_sampler(shape, device, z_type):
    if z_type == 'gaussian':
        z = torch.randn(shape, device=device)
    elif z_type == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z    # [N, 512]


def val_style_sampler(batch_size, img_size):
    dataset = datasets.AAHQ('data/aahq', 'style_codes.csv', img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False, num_workers=4)
    for x_b, z_b in dataloader:
        fixed_x_b = x_b
        fixed_z_b = z_b
        break   
    return fixed_x_b, fixed_z_b    # [25, 3, 64, 64], [25, 512]


def init_queue(batch_size, img_size):
    dataset = datasets.AAHQ('data/aahq', 'style_codes.csv', img_size)
    # shuffle=True, 每次迭代dataloader之前数据都会被shuffle
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False, num_workers=4)
    for _, z_b in dataloader:
        fixed_z_b = z_b
        break
    z_b_queue = fixed_z_b
    return z_b_queue    # [12, 512]


def transform(x, size):
    trans = transforms.Compose([transforms.Resize((size, size))])
    return trans(x)


def train(rank, world_size, opt):
    # 初始化
    torch.cuda.set_device(rank)
    setup(rank, world_size, opt.port)
    device = torch.device(rank)
    # torch.backends.cudnn.benchmark = False
    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    # 25个固定的隐变量z，用于生成验证结果, 验证时返回5x5个人脸
    fixed_z_a = z_sampler((25, 512), device='cpu', z_type=metadata['z_dist'])
    fixed_x_b, fixed_z_b = val_style_sampler(25, metadata['img_size_sr'])

    # 实例化SIREN模型
    FiLM = getattr(film_artnerf, metadata['model'])
    # 缩短训练时间、降低存储需求
    scaler = torch.cuda.amp.GradScaler()
    
    # 加载模型
    if opt.load_dir != '':
        generator = torch.load(os.path.join(opt.load_dir, '00_300000_generator.pth'), map_location=device)
        # generator.siren.wbm.w.data = (torch.ones(9 + 2, 1) * 0.7).to(device)
        generator.siren.wbm.w.requires_grad = True
        discriminator_real = torch.load(os.path.join(opt.load_dir, '00_300000_discriminator_real.pth'), map_location=device)
        # discriminator_style = torch.load(os.path.join(opt.load_dir, 'discriminator_style.pth'), map_location=device)
        # discriminator_latent = torch.load(os.path.join(opt.load_dir, 'discriminator_latent.pth'), map_location=device)
        discriminator_style = getattr(discriminators, metadata['discriminator_style'])().to(device)
        discriminator_latent = getattr(discriminators, metadata['discriminator_latent'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        # 若是加载预训练模型, 这里需要注释掉(因为只需要模型参数, 不需要状态参数), 若是继续训练预训练模型, 这里无需注释
        # ema.load_state_dict(torch.load(os.path.join(opt.load_dir, "ema.pth"), map_location=device))
    else:
        generator = getattr(generators_artnerf, metadata['generator'])(FiLM, metadata['z_dim'], metadata['hidden_dim']).to(device)
        discriminator_real = getattr(discriminators, metadata['discriminator_real'])().to(device)
        discriminator_style = getattr(discriminators, metadata['discriminator_style'])().to(device)
        discriminator_latent = getattr(discriminators, metadata['discriminator_latent'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    
    # Maybe we don't need find_unused_parameters=True
    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_real_ddp = DDP(discriminator_real, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    discriminator_style_ddp = DDP(discriminator_style, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    discriminator_latent_ddp = DDP(discriminator_latent, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)

    # 若unique_lr=True, 则为mapping_network单独设置lr, 否则整个G用同一个lr
    if metadata.get('unique_lr', False):
        content_mapping_network_param_names = [n for n, _ in generator_ddp.module.siren.content_mapping_network.named_parameters()]
        style_mapping_network_param_names = [n for n, _ in generator_ddp.module.siren.style_mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in content_mapping_network_param_names + style_mapping_network_param_names]
        # generator_parameters是除了mapping_network之外的参数
        generator_parameters = [p for n, p in generator_ddp.module.named_parameters() if n not in content_mapping_network_param_names + style_mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr': metadata['gen_lr']*5e-2}],
                                        lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        wbm_parameters = [p for p in generator_ddp.module.siren.wbm.parameters()]
        generator_parameters = [p for n, p in generator_ddp.module.named_parameters() if n != 'siren.wbm.w']
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': wbm_parameters, 'name': 'wbm', 'lr': metadata['wbm_lr']}],
                                        lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam([{'params': list(discriminator_real_ddp.parameters()), 'name': 'discriminator_real'},
                                    {'params': list(discriminator_style_ddp.parameters()), 'name': 'discriminator_style', 'lr': metadata['dis_style_lr']},
                                    {'params': list(discriminator_latent_ddp.parameters()), 'name': 'discriminator_latent', 'lr': metadata['dis_latent_lr']}],
                                    lr=metadata['dis_real_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    # 若是加载预训练模型, 这里需要注释掉, 若是继续训练模型, 这里无需注释掉
    # if opt.load_dir != '':
    #     optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth')))
    #     optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth')))
    #     if not metadata.get('disable_scaler', False):
    #         scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scaler.pth')))

    generator_losses = []
    discriminator_losses = []

    cur_epoch = 0
    cur_step = 0
    if opt.set_step != None:
        cur_step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator_ddp.module.set_device(device)

    # ----------------------------------------------------------------------------
    #  Training
    # logging.basicConfig(filename=os.path.join(opt.output_dir, 'logs.txt'), level=logging.DEBUG)
    writer = SummaryWriter(os.path.join(opt.output_dir, 'tb_logs_' + str(datetime.now())[5:10]))

    with open(os.path.join(opt.output_dir, 'options_' + str(datetime.now())[5:19] + '.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator_ddp.module))
        f.write('\n\n')
        f.write(str(discriminator_real_ddp.module))
        f.write('\n\n')
        f.write(str(discriminator_style_ddp.module))
        f.write('\n\n')
        f.write(str(discriminator_latent_ddp.module))
        f.write('\n\n')
        f.write(str(curriculum))
        
    # torch.manual_seed(rank)

    dataloader1 = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(cur_epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)

    # init embedding_queue
    embedding_queue = torch.empty(0, metadata['z_dim']).to(device)
    z_b_queue1 = init_queue(metadata['batch_size'], metadata['img_size_sr']).to(device)  # [12, 512]
    z_b_queue2 = init_queue(metadata['batch_size'], metadata['img_size_sr']).to(device)  # [12, 512]

    # main epoch loop
    for epoch in range(opt.n_epochs):
        total_progress_bar.update(1)
        metadata = curriculums.extract_metadata(curriculum, cur_step)

        if not dataloader1 or dataloader1.batch_size != metadata['batch_size']:
            dataloader1 = datasets.get_dataset(metadata['dataset_artnerf'], **metadata)
            step_next_upsample = curriculums.next_upsample_step(curriculum, cur_step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, cur_step)
            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((cur_step - step_last_upsample))

        for i, (x_a, x_b_128, z_b) in enumerate(dataloader1):
            metadata = curriculums.extract_metadata(curriculum, cur_step)
            
            if dataloader1.batch_size != metadata['batch_size']:
                break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            # alpha是渐进式D的fade in过程的调节权重 
            alpha = min(1, (cur_step - step_last_upsample) / (metadata['fade_steps']))
            # not important                                                                     
            metadata['nerf_noise'] = max(0, 1. - cur_step/5000.)

            # 便于某次eval后切换为train
            generator_ddp.train()
            discriminator_real_ddp.train()
            discriminator_style_ddp.train()
            discriminator_latent_ddp.train()

            z_b = z_b.to(device)            # [12, 512]
            z_b_copy = copy.deepcopy(z_b)   # [12, 512]
            x_a = x_a.to(device)            # [12, 3, 128, 128]
            x_b_128 = x_b_128.to(device)    # [12, 3, 128, 128]

            # 固定G, 训练D
            with torch.cuda.amp.autocast():
                # 截断bp流，计算完loss关于D的所有参数的偏导后就不再继续往后进行, 这么做会极大地节省显存
                with torch.no_grad():  # with torch.no_grad()内的每个张量都将requires_grad设置为False
                    z_a = z_sampler((z_b.shape[0], metadata['z_dim']), device=device, z_type=metadata['z_dist'])
                    split_batch_size = z_a.shape[0] // metadata['batch_split']
                    fake_a_sr = []
                    fake_b_sr = []
                    fake_a_pos = []
                    fake_b_pos = []
                    for split in range(metadata['batch_split']):
                        subset_z_a = z_a[split * split_batch_size : (split+1) * split_batch_size]
                        subset_z_b = z_b[split * split_batch_size : (split+1) * split_batch_size]
                        # generate fake_a and fake_b
                        _, subset_fake_a_sr, subset_fake_a_pos = generator_ddp(11, subset_z_a, None, **metadata)
                        _, subset_fake_b_sr, subset_fake_b_pos = generator_ddp(0, subset_z_a, subset_z_b, **metadata)
                        fake_a_sr.append(subset_fake_a_sr)
                        fake_b_sr.append(subset_fake_b_sr)
                        fake_a_pos.append(subset_fake_a_pos)
                        fake_b_pos.append(subset_fake_b_pos)
                    fake_a_sr = torch.cat(fake_a_sr, axis=0)      # [12, 3, 128, 128]
                    fake_b_sr = torch.cat(fake_b_sr, axis=0)      # [12, 3, 128, 128]
                    fake_a_pos = torch.cat(fake_a_pos, axis=0)    # [12, 2]
                    fake_b_pos = torch.cat(fake_b_pos, axis=0)    # [12, 2]
                    # generate fake_b_queue1
                    _, fake_b_queue1, _ = generator_ddp(0, z_a, z_b_queue1, **metadata)  # [12, 3, 128, 128]

                x_a.requires_grad = True
                x_b_128.requires_grad = True
                z_b.requires_grad = True

                # real_a -> D_real
                real_a_pred, _ = discriminator_real_ddp(x_a, alpha, **metadata)
                # real_b -> D_style
                real_b_pred, _ = discriminator_style_ddp(x_b_128, alpha, **metadata)
                # (real_b, latent_b) -> D_latent
                real_style_pred = discriminator_latent_ddp(x_b_128, z_b, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # calculate grad
                inv_scale = 1./scaler.get_scale()
                grad_real_a = torch.autograd.grad(outputs=scaler.scale(real_a_pred.sum()), inputs=x_a, create_graph=True)
                grad_real_a = [p * inv_scale for p in grad_real_a][0]
                grad_real_b = torch.autograd.grad(outputs=scaler.scale(real_b_pred.sum()), inputs=x_b_128, create_graph=True)
                grad_real_b = [p * inv_scale for p in grad_real_b][0]
                grad_real_style1 = torch.autograd.grad(outputs=scaler.scale(real_style_pred.sum()), inputs=x_b_128, create_graph=True)
                grad_real_style1 = [p * inv_scale for p in grad_real_style1][0]
                grad_real_style2 = torch.autograd.grad(outputs=scaler.scale(real_style_pred.sum()), inputs=z_b, create_graph=True)
                grad_real_style2 = [p * inv_scale for p in grad_real_style2][0]

            with torch.cuda.amp.autocast():
                # grad_penalty
                if metadata['r1_lambda'] > 0:
                    grad_penalty_a = (grad_real_a.view(grad_real_a.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty_b = (grad_real_b.view(grad_real_b.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty_style1 = (grad_real_style1.view(grad_real_style1.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty_style2 = (grad_real_style2.view(grad_real_style2.size(0), -1).norm(2, dim=1) ** 2).mean()
                else:
                    grad_penalty = 0

                # fake_a -> D_real
                fake_a_pred, fake_a_pos_pred = discriminator_real_ddp(fake_a_sr, alpha, **metadata)
                # fake_b -> D_style
                fake_b_pred, fake_b_pos_pred = discriminator_style_ddp(fake_b_sr, alpha, **metadata)
                # (fake_b, random_style_latent) -> D_latent
                fake_style_pred = discriminator_latent_ddp(fake_b_queue1, z_b_queue1, alpha, **metadata)

                # calculate loss for each D
                d_real_loss = dis_loss(fake_a_pred, real_a_pred) + pos_loss(fake_a_pos, fake_a_pos_pred) * metadata['pos_lambda'] + 0.5 * grad_penalty_a * metadata['r1_lambda']
                d_style_loss = dis_loss(fake_b_pred, real_b_pred) + pos_loss(fake_b_pos, fake_b_pos_pred) * metadata['pos_lambda'] + 0.5 * grad_penalty_b * metadata['r1_lambda']
                d_latent_loss = dis_loss(fake_style_pred, real_style_pred) + 0.5 * (grad_penalty_style1 + grad_penalty_style2) * metadata['r1_lambda']
                d_latent_loss = dis_loss(fake_style_pred, real_style_pred)

                # total d_loss
                # d_loss = d_real_loss + d_style_loss
                d_loss = d_real_loss + d_style_loss + d_latent_loss
                discriminator_losses.append(d_loss.item())
            
            # 混合精度的梯度反传及参数更新写法
            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_real_ddp.parameters(), metadata['grad_clip'])
            torch.nn.utils.clip_grad_norm_(discriminator_style_ddp.parameters(), metadata['grad_clip'])
            torch.nn.utils.clip_grad_norm_(discriminator_latent_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)

            ###################################################################################################
            # 固定D, 训练G
            # 对于生成器而言，我们的确计算了D的梯度，但是我们没有更新D的权重（只写了optimizer_G.step），所以训练生成器的时候也就不会改变判别器了
            lambda_1 = 0.2
            lambda_2 = 0.5
            lambda_3 = 0.3
            z_a = z_sampler((z_b.shape[0], metadata['z_dim']), device=device, z_type=metadata['z_dist'])
            split_batch_size = z_a.shape[0] // metadata['batch_split']
            for split in range(metadata['batch_split']):
                # prepare data
                subset_z_a = z_a[split * split_batch_size : (split+1) * split_batch_size]               # [4, 512]
                subset_z_b = z_b[split * split_batch_size : (split+1) * split_batch_size]               # [4, 512]
                subset_z_b_queue2 = z_b_queue2[split * split_batch_size : (split+1) * split_batch_size] # [4, 512]
                
                # generate fake_a
                with torch.cuda.amp.autocast():
                    _, fake_a_sr, fake_a_pos = generator_ddp(11, subset_z_a, None, **metadata)
                    # fake_a -> D_real
                    fake_a_pred, fake_a_pos_pred = discriminator_real_ddp(fake_a_sr, alpha, **metadata)
                    # topk_percentage = max(0.99 ** (cur_step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_percentage = 1.0
                    topk_num = math.ceil(topk_percentage * fake_a_pred.shape[0])
                    fake_a_pred = torch.topk(fake_a_pred, topk_num, dim=0).values
                    # calculate fake_a_adv_loss
                    fake_a_adv_loss = gen_loss(fake_a_pred) * metadata['adv_lambda']
                    # calculate pos loss for fake_a
                    fake_a_pos_loss = torch.nn.MSELoss()(fake_a_pos, fake_a_pos_pred) * metadata['pos_lambda']
                    fake_a_loss = lambda_1 * (fake_a_adv_loss + fake_a_pos_loss)

                # 执行scaler.scale(d_loss).backward()后, intermediate variables会被释放, 导致大量显存释放
                scaler.scale(fake_a_loss).backward()

                # generate fake_b
                with torch.cuda.amp.autocast():
                    _, fake_b_sr, fake_b_pos = generator_ddp(0, subset_z_a, subset_z_b, **metadata)
                    # fake_b -> D_style
                    fake_b_pred, fake_b_pos_pred = discriminator_style_ddp(fake_b_sr, alpha, **metadata)
                    fake_b_pred = torch.topk(fake_b_pred, topk_num, dim=0).values
                    # calculate fake_b_adv_loss
                    fake_b_adv_loss = gen_loss(fake_b_pred) * metadata['adv_lambda']
                    # calculate pos loss for fake_b
                    fake_b_pos_loss = torch.nn.MSELoss()(fake_b_pos, fake_b_pos_pred) * metadata['pos_lambda']
                    # fake_b_loss = fake_b_adv_loss + fake_b_pos_loss
                    fake_b_loss = lambda_2 * (fake_b_adv_loss + fake_b_pos_loss)

                scaler.scale(fake_b_loss).backward()

                # generate fake_b_with_queue
                with torch.cuda.amp.autocast():
                    _, fake_b_queue2, _ = generator_ddp(0, subset_z_a, subset_z_b_queue2, **metadata)
                    # (fake_b_with_queue, random_style_latent) -> D_latent
                    fake_style_pred = discriminator_latent_ddp(fake_b_queue2, subset_z_b_queue2, alpha, **metadata)
                    fake_style_pred = torch.topk(fake_style_pred, topk_num, dim=0).values
                    # calculate fake_style_adv_loss
                    fake_style_adv_loss = gen_loss(fake_style_pred) * metadata['adv_lambda']
                    # fake_style_loss = fake_style_adv_loss
                    fake_style_loss = lambda_3 * fake_style_adv_loss

                scaler.scale(fake_style_loss).backward()
                
                # total loss
                g_loss = fake_a_loss + fake_b_loss + fake_style_loss
                generator_losses.append(g_loss.item())
            
            # 混合精度的梯度反传及参数更新写法
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            generator_ddp.module.siren.wbm.w.data = generator_ddp.module.siren.wbm.w.data.clamp(0.0, 1.0)
            ema.update(generator_ddp.parameters())

            # 若z_b_queue1 = z_b[idx1], 则会在下一个batch训练时第二次对z_b计算梯度, 导致报错, 因此需要deepcopy
            if epoch == 0:
                embedding_queue = torch.cat([embedding_queue, z_b_copy], axis=0)
            
            idx1 = torch.tensor(random.sample(range(embedding_queue.shape[0]), z_b.shape[0])).to(device)
            idx2 = torch.tensor(random.sample(range(embedding_queue.shape[0]), z_b.shape[0])).to(device)
            z_b_queue1 = torch.index_select(embedding_queue, 0, idx1)
            z_b_queue2 = torch.index_select(embedding_queue, 0, idx2)


            # 只需在第一台机器第一张卡上面保存模型和生成log
            if rank == 0:
                # update子进度条
                interior_step_bar.update(1)
                writer.add_scalars("loss_D", {"d_real_loss": d_real_loss, "d_style_loss": d_style_loss, "d_latent_loss": 0}, cur_step)
                writer.add_scalars("loss_G", {"fake_a_loss": (1/lambda_1)*fake_a_loss, "fake_b_loss": (1/lambda_2)*fake_b_loss, "fake_style_loss": (1/lambda_3)*0}, cur_step)
                w = generator_ddp.module.siren.wbm.w
                writer.add_scalars("WBM's weights", {"w0": w[0], "w1": w[1], "w2": w[2], "w3": w[3], "w4": w[4], "w5": w[5], "w6": w[6], "w7": w[7], "w8": w[8], "w9": w[9], "w10": w[10]}, cur_step)
                # tqdm.write(f"GPU cost: {torch.cuda.memory_allocated()/1024/1024/1024}")
                
                # 单个epoch内，每隔50个batch输出一次log
                if i % 10 == 0:
                    tqdm.write(f"[Experiment: {opt.output_dir}] [Epoch: {cur_epoch}/{opt.n_epochs}] [Step: {cur_step}] [loss_d1: {d_real_loss.item()}] [loss_d2: {d_style_loss.item()}] [loss_d3: {0}] [loss_g1: {(1/lambda_1)*fake_a_loss.item()}] [loss_g2: {(1/lambda_2)*fake_b_loss.item()}] [loss_g3: {(1/lambda_3)*0}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]")
                    tqdm.write(f"wbm's weight: {generator_ddp.module.siren.wbm.w.detach().cpu().reshape(1, 11)[0]}")
                  
                if cur_step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    # 存一下固定的B域图像
                    save_image(fixed_x_b.to(device)[:20], os.path.join(opt.output_dir, f"style_imgs.png"), nrow=5, normalize=True)
                    # 输出固定的5x5图像结果
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0 # 标准差为0, 所有视角均为固定
                            copied_metadata['img_size'] = 32
                            copied_metadata['img_size_sr'] = 128
                            fake_a_imgs, fake_a_imgs_sr, _ = generator_ddp.module.staged_forward(11, fixed_z_a.to(device), None, **copied_metadata)
                            fake_b_imgs, fake_b_imgs_sr, _ = generator_ddp.module.staged_forward(0, fixed_z_a.to(device), fixed_z_b.to(device), **copied_metadata)
                    save_image(fake_a_imgs[:20], os.path.join(opt.output_dir, f"fixed_fake_a.png"), nrow=5, normalize=True)
                    save_image(fake_a_imgs_sr[:20], os.path.join(opt.output_dir, f"fixed_fake_a_sr.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs[:20], os.path.join(opt.output_dir, f"fixed_fake_b.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs_sr[:20], os.path.join(opt.output_dir, f"fixed_fake_b_sr.png"), nrow=5, normalize=True)
                    
                    # 输出倾斜视角的5x5图像结果
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5    # 调整Mean of camera yaw以改变生成图像的视角
                            copied_metadata['img_size'] = 32
                            copied_metadata['img_size_sr'] = 128
                            fake_a_imgs, fake_a_imgs_sr, _ = generator_ddp.module.staged_forward(11, fixed_z_a.to(device), None, **copied_metadata)
                            fake_b_imgs, fake_b_imgs_sr, _ = generator_ddp.module.staged_forward(0, fixed_z_a.to(device), fixed_z_b.to(device), **copied_metadata)
                    save_image(fake_a_imgs[:20], os.path.join(opt.output_dir, f"tilted_fake_a.png"), nrow=5, normalize=True)
                    save_image(fake_a_imgs_sr[:20], os.path.join(opt.output_dir, f"tilted_fake_a_sr.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs[:20], os.path.join(opt.output_dir, f"tilted_fake_b.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs_sr[:20], os.path.join(opt.output_dir, f"tilted_fake_b_sr.png"), nrow=5, normalize=True)
                    
                    ema.store(generator_ddp.parameters())    # 存一下当前参数
                    ema.copy_to(generator_ddp.parameters())  # 加载ema参数
                    generator_ddp.eval()
                    # 输出ema下固定的5x5图像结果
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 32
                            copied_metadata['img_size_sr'] = 128
                            fake_a_imgs, fake_a_imgs_sr, _ = generator_ddp.module.staged_forward(11, fixed_z_a.to(device), None, **copied_metadata)
                            fake_b_imgs, fake_b_imgs_sr, _ = generator_ddp.module.staged_forward(0, fixed_z_a.to(device), fixed_z_b.to(device), **copied_metadata)
                    save_image(fake_a_imgs[:20], os.path.join(opt.output_dir, f"ema_fixed_fake_a.png"), nrow=5, normalize=True)
                    save_image(fake_a_imgs_sr[:20], os.path.join(opt.output_dir, f"ema_fixed_fake_a_sr.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs[:20], os.path.join(opt.output_dir, f"ema_fixed_fake_b.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs_sr[:20], os.path.join(opt.output_dir, f"ema_fixed_fake_b_sr.png"), nrow=5, normalize=True)
                    
                    # 输出ema下倾斜视角的5x5图像结果
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 32
                            copied_metadata['img_size_sr'] = 128
                            fake_a_imgs, fake_a_imgs_sr, _ = generator_ddp.module.staged_forward(11, fixed_z_a.to(device), None, **copied_metadata)
                            fake_b_imgs, fake_b_imgs_sr, _ = generator_ddp.module.staged_forward(0, fixed_z_a.to(device), fixed_z_b.to(device), **copied_metadata)
                    save_image(fake_a_imgs[:20], os.path.join(opt.output_dir, f"ema_tilted_fake_a.png"), nrow=5, normalize=True)
                    save_image(fake_a_imgs_sr[:20], os.path.join(opt.output_dir, f"ema_tilted_fake_a_sr.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs[:20], os.path.join(opt.output_dir, f"ema_tilted_fake_b.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs_sr[:20], os.path.join(opt.output_dir, f"ema_tilted_fake_b_sr.png"), nrow=5, normalize=True)
                           
                    # 输出z为随机数, x_b固定的5x5图像结果
                    rand_z_a = torch.randn_like(fixed_z_a).to(device)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            copied_metadata['img_size'] = 32
                            copied_metadata['img_size_sr'] = 128
                            fake_a_imgs, fake_a_imgs_sr, _ = generator_ddp.module.staged_forward(11, rand_z_a, None, **copied_metadata)
                            fake_b_imgs, fake_b_imgs_sr, _ = generator_ddp.module.staged_forward(0, rand_z_a, fixed_z_b.to(device), **copied_metadata)
                    save_image(fake_a_imgs[:20], os.path.join(opt.output_dir, f"random_z_fixed_fake_a.png"), nrow=5, normalize=True)
                    save_image(fake_a_imgs_sr[:20], os.path.join(opt.output_dir, f"random_z_fixed_fake_a_sr.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs[:20], os.path.join(opt.output_dir, f"random_z_fixed_fake_b.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs_sr[:20], os.path.join(opt.output_dir, f"random_z_fixed_fake_b_sr.png"), nrow=5, normalize=True)
                    
                    # 输出z为随机数, x_b固定的倾斜视角5x5图像结果
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 32
                            copied_metadata['img_size_sr'] = 128
                            fake_a_imgs, fake_a_imgs_sr, _ = generator_ddp.module.staged_forward(11, rand_z_a, None, **copied_metadata)
                            fake_b_imgs, fake_b_imgs_sr, _ = generator_ddp.module.staged_forward(0, rand_z_a, fixed_z_b.to(device), **copied_metadata)
                    save_image(fake_a_imgs[:20], os.path.join(opt.output_dir, f"random_z_tilted_fake_a.png"), nrow=5, normalize=True)
                    save_image(fake_a_imgs_sr[:20], os.path.join(opt.output_dir, f"random_z_tilted_fake_a_sr.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs[:20], os.path.join(opt.output_dir, f"random_z_tilted_fake_b.png"), nrow=5, normalize=True)
                    save_image(fake_b_imgs_sr[:20], os.path.join(opt.output_dir, f"random_z_tilted_fake_b_sr.png"), nrow=5, normalize=True)

                    # 恢复当前的参数, 以便后续训练
                    ema.restore(generator_ddp.parameters())
                    # 每 2000 step 保存一次最新的模型
                    torch.save(ema.state_dict(), os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(generator_ddp.module, os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_real_ddp.module, os.path.join(opt.output_dir, 'discriminator_real.pth'))
                    torch.save(discriminator_style_ddp.module, os.path.join(opt.output_dir, 'discriminator_style.pth'))
                    torch.save(discriminator_latent_ddp.module, os.path.join(opt.output_dir, 'discriminator_latent.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))

            if opt.fid_interval > 0 and (cur_step + 1) % opt.fid_interval == 0:
                copied_metadata = copy.deepcopy(metadata)
                copied_metadata['img_size'] = 32
                copied_metadata['img_size_sr'] = 128
                generated_dir1 = os.path.join(opt.output_dir, 'evaluation/generated_a')
                generated_dir2 = os.path.join(opt.output_dir, 'evaluation/generated_b')
                if rank == 0:
                    fid_evaluation.setup_evaluation1(metadata['dataset_pigan'], generated_dir1, dataset_path=metadata["dataset_path"], target_size=copied_metadata['img_size_sr'])
                    fid_evaluation.setup_evaluation2(metadata['dataset_aahq'], generated_dir2, dataset_path2=metadata["dataset_path2"], dataset_path3=metadata["dataset_path3"], target_size=copied_metadata['img_size_sr'])
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images1(generator_ddp, copied_metadata['img_size'], copied_metadata['img_size_sr'], metadata, rank, world_size, generated_dir1)
                fid_evaluation.output_images2(generator_ddp, copied_metadata['img_size'], copied_metadata['img_size_sr'], metadata, rank, world_size, generated_dir2)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                torch.cuda.empty_cache()

            cur_step += 1  # 处理完一个batch, step+1
        cur_epoch += 1  # 处理完一个epoch, epoch+1

    writer.close()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='16666', help="different training process should use different ports")
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument('--set_step', type=int, default=None, help="set the step of current training")
    parser.add_argument('--load_dir', type=str, default='', help="directory of generator.pth")
    parser.add_argument('--curriculum', type=str, required=True, help="config file")
    parser.add_argument('--output_dir', type=str, default='out_dir', help="where to place outputs")
    parser.add_argument("--sample_interval", type=int, default=2000, help="interval between validating the model")
    parser.add_argument('--model_save_interval', type=int, default=6000, help="interval between saving trained models")
    parser.add_argument('--fid_interval', type=int, default=2000, help="interval between evaluating fid socre")

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))  # 即world_size
    # 启动多个进程, 进程个数和GPU个数一致
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
