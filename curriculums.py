import math

def next_upsample_step(curriculum, current_step):
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step].get('img_size', 512) > current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

# curriculum是字典，如CelebA
def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


facestylization = {
    # 学习率视情况而定,一般batch_size越大, lr需成比例增大
    # train with a small dataset might require more epochs than large datasets.
    # If you're looking for good shapes, e.g. for CelebA, try increasing num_steps and moving the back plane (ray_end) to allow the model to move the background back and capture the full head.
    # If your batch size is small, use stronger R1 regularization
    # The strength of R1 regularization is an important hyperparameter for ensuring stability of GAN training. The best value of gamma may vary widely between datasets. If you have nothing to go on, --gamma=5 is a safe choice. If training seems stable, and your model starts to produce diverse and reasonable outputs, you can try lowering gamma. If you experience training instability or mode collapse, try increasing gamma. In general, if your batch size is small, or if your images are large, you will need more regularization (higher gamma).
    # 这边的batch_size是单卡的batch_size
    0: {'batch_size': 8, 'num_steps': 24, 'img_size': 32, 'img_size_sr': 128, 'batch_split': 2, 'gen_lr': 4e-5, 'wbm_lr': 1e-2, 'dis_real_lr': 1e-4, 'dis_style_lr': 1e-4,'dis_latent_lr': 3e-5, 'r1_lambda': 0.2},
    int(200000): {'batch_size': 8, 'num_steps': 12, 'img_size': 256, 'batch_split': 4, 'gen_lr': 4e-5, 'wbm_lr': 1e-3, 'dis_real_lr': 4e-5, 'dis_style_lr': 4e-5,'dis_latent_lr': 1e-5},
    int(500000): {},
    'dataset_pigan': 'CelebA',
    'dataset_aahq': 'AAHQ',
    'dataset_artnerf': 'FaceStylization',
    'refer_dataset': 'referFace',
    'dataset1': 'realFace',
    'dataset2': 'aniFace',
    'dataset_path': 'data/celeba',
    'dataset_path1': 'data/celeba_mini',
    'dataset_path2': 'data/aahq',
    'dataset_path3': 'style_codes.csv',
    'mode': 'train',
    'nerf_random': False,
    'hierarchical_sample': True,
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'z_dim': 512,
    'style_dim': 256,
    'f_dim': 128,
    'z_dist': 'gaussian',
    'hidden_dim': 256,
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'model': 'FiLMSirenArtNeRF',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'discriminator_u': 'D_U',
    'discriminator_x': 'D_X',
    'discriminator_y': 'D_Y',
    'discriminator_real': 'CCSEncoderDiscriminator_artnerf',
    'discriminator_style': 'CCSEncoderDiscriminator_artnerf',
    'discriminator_latent': 'ProjectionDiscriminator',
    'neural_renderer': 'NeuralRenderer',
    'is_dual_disc': False,
    'perc_backbone': 'vgg19',
    'clamp_mode': 'relu',
    'z_lambda': 0,
    'adv_lambda': 1,
    'r1_lambda': 0.1,
    'grad_clip': 10,
    'pos_lambda': 5,
    'last_back': False,
    'eval_last_back': True,
}