# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../' )
import random

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.realestate10k_dataio import RealEstate10kVis, get_camera_pose
import torch
import models
import training
import configargparse
from torch.utils.data import DataLoader
import loss_functions
import summaries
from utils import util
import config
from tqdm import tqdm
from imageio import imwrite, get_writer
from glob import glob
import time
import matplotlib.pyplot as plt

import lpips
from skimage.metrics import structural_similarity
import config

# torch.manual_seed(0)
# python experiment_scripts/eval_realestate10k.py --experiment_name vis_realestate --batch_size 2 --gpus 2 --views 2 --checkpoint_path realestate_query_1/64_2_512/checkpoints/model_current.pth

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))


def make_circle(n, radius=0.1):
    angles = np.linspace(0, 4 * np.pi, n)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius

    coord = np.stack([x, y, np.zeros(n)], axis=-1)
    return coord


def multigpu_train(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)

    val_dataset = RealEstate10kVis(img_root="/home/dev4/data/SKY/datasets/data_download/realestate/test",
                                pose_root="/home/dev4/data/SKY/datasets/poses/realestate/test.mat",
                                num_ctxt_views=opt.views, num_query_views=1, augment=False)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn)

    #model = models.CrossAttentionRenderer(no_multiview=opt.no_multiview, no_sample=opt.no_sample, no_latent_concat=opt.no_latent_concat, model=opt.model, n_view=opt.views)
    model = models.CrossAttentionRenderer(no_sample=opt.no_sample, 
                                            no_latent_concat=opt.no_latent_concat,
                                            no_multiview=opt.no_multiview,
                                            no_high_freq=opt.no_high_freq, 
                                            model=opt.model, 
                                            n_view=opt.views, 
                                            )
    old_state_dict = model.state_dict()

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        if opt.reconstruct:
            state_dict['latent_codes.weight'] = torch.zeros_like(state_dict['latent_codes.weight'])

        # state_dict['encoder.latent'] = old_state_dict['encoder.latent']

        model.load_state_dict(state_dict['model'], strict=False)

    model = model.cuda().eval()
    device = "gpu"

    total_mse, total_psnr, total_lpips, total_ssim = [],[],[],[] 
    with torch.no_grad():
        loss_fn_alex = lpips.LPIPS(net='vgg').cuda()

        mses = []
        psnrs = []
        lpips_list = []
        ssims = []

        for val_i, (model_input, gt) in enumerate(val_loader):
            if device == 'gpu':
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

            model_input_full = model_input
            rgb_full = model_input['query']['rgb']
            uv_full = model_input['query']['uv']

            nrays = uv_full.size(2)

            if  opt.model == 'query':
                z, _ = model.get_z(model_input)
            else:
                z = model.get_z(model_input)

            if opt.views == 3:
                rgb_chunks = torch.chunk(rgb_full, 18, dim=2)
                uv_chunks = torch.chunk(uv_full, 18, dim=2)
            else:
                rgb_chunks = torch.chunk(rgb_full, 9, dim=2)
                uv_chunks = torch.chunk(uv_full, 9, dim=2)

            start = time.time()

            model_outputs = []

            for rgb_chunk, uv_chunk in zip(rgb_chunks, uv_chunks):
                model_input['query']['rgb'] = rgb_chunk
                model_input['query']['uv'] = uv_chunk

                model_output = model(model_input, z=z, val=True)

                model_outputs.append(model_output)

            end = time.time()
            print("elapsed: ", end - start)
            model_output_full = {}
            for k in ['rgb', 'valid_mask', 'depth_ray']:
                outputs = [model_output[k] for model_output in model_outputs]

                if k == "pixel_val":
                    val = torch.cat(outputs, dim=-3)
                else:
                    val = torch.cat(outputs, dim=-2)
                model_output_full[k] = val

            rgb = model_output_full['rgb'].view(256, 256, 3)
            valid_mask = model_output_full['valid_mask'].view(256, 256, 1)

            # Saving output image
            target = gt['rgb'].view(256, 256, 3)

            rgb = ((rgb + 1) * 0.5).detach() * valid_mask + 0.5 * (1 - valid_mask) * torch.ones_like(rgb)
            target = ((target + 1) * 0.5).detach() * valid_mask + 0.5 * (1 - valid_mask) * torch.ones_like(target)

            mse = img2mse(rgb, target)
            psnr = mse2psnr(mse)
            mses.append(mse.item())
            psnrs.append(psnr.item())

            rgb_lpips = ((rgb.permute(2, 0, 1) - 0.5) * 2)[None, :, :, :].cuda()
            target_lpips = ((target.permute(2, 0, 1) - 0.5) * 2)[None, :, :, :].cuda()
            lpip = loss_fn_alex(rgb_lpips, target_lpips).item()
            lpips_list.append(lpip)

            rgb_np = rgb.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            ssim = structural_similarity(rgb_np, target_np, win_size=11, multichannel=True, gaussian_weights=True)
            ssims.append(ssim)

            print("mse, psnr, lpip, ssim", np.mean(mses), np.mean(psnrs), np.mean(lpips_list), np.mean(ssims))

        f = os.path.join(opt.logging_root, 'eval.txt')
        with open(f, 'w') as file:
            file.write('{} = {}\n'.format('MSE', np.mean(mses)))
            file.write('{} = {}\n'.format('PSNR', np.mean(psnrs)))
            file.write('{} = {}\n'.format('LPIPS', np.mean(lpips_list)))
            file.write('{} = {}\n'.format('SSIM', np.mean(ssims)))


if __name__ == "__main__":
    opt = config.config_parser()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)