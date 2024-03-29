import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import random

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.realestate10k_dataio import RealEstate10k
import torch
import models
import training
from torch.utils.data import DataLoader
import loss_functions
import summaries
import config

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))


def multigpu_train(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1493', world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)

    #
    basedir = os.path.join(opt.logging_root, opt.experiment_name)
    os.makedirs(basedir, exist_ok=True)
    f = os.path.join(basedir, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(opt):
            attr = getattr(opt, arg)
            file.write('{} = {}\n'.format(arg, attr))

    def create_dataloader_callback(sidelength, batch_size, query_sparsity):
        train_dataset = RealEstate10k(img_root="/home/dev4/data/SKY/datasets/data_download/realestate/train",
                                     pose_root="/home/dev4/data/SKY/datasets/poses/realestate/train.mat",
                                     num_ctxt_views=opt.views, num_query_views=1, query_sparsity=192,
                                     lpips=opt.lpips, augment=(not opt.no_data_aug))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)

        val_dataset = RealEstate10k(img_root="/home/dev4/data/SKY/datasets/data_download/realestate/test",
                                      pose_root="/home/dev4/data/SKY/datasets/poses/realestate/test.mat",
                                      num_ctxt_views=opt.views, num_query_views=1, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)

        return train_loader, val_loader

    model = models.CrossAttentionRenderer(no_sample=opt.no_sample, 
                                          no_latent_concat=opt.no_latent_concat,
                                          no_multiview=opt.no_multiview,
                                          no_high_freq=opt.no_high_freq, 
                                          model=opt.model, 
                                          n_view=opt.views, 
                                          num_queries=opt.num_queries,
                                          contra_loss = opt.contra
                                          )
    optimizer = torch.optim.Adam(lr=opt.lr, params=model.parameters(), betas=(0.99, 0.999))

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        state_dict, optimizer_dict = state_dict['model'], state_dict['optimizer']

        model.load_state_dict(state_dict, strict=False)
        # optimizer.load_state_dict(optimizer_dict)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    model = model.cuda()

    if opt.gpus > 1:
        sync_model(model)

    # Define the loss
    summary_fn = summaries.img_summaries
    val_summary_fn = summaries.img_summaries
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    loss_fn = val_loss_fn = loss_functions.LFLoss(opt.l2_coeff, opt.lpips, opt.depth, opt.contra)

    training.training(model=model, dataloader_callback=create_dataloader_callback,
                                 dataloader_iters=(1000000,), dataloader_params=((64, opt.batch_size, 512), ),
                                 epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                                 epochs_til_checkpoint=opt.epochs_til_ckpt,
                                 model_dir=root_path, loss_fn=loss_fn, val_loss_fn=val_loss_fn,
                                 iters_til_checkpoint=opt.iters_til_ckpt, val_summary_fn=val_summary_fn,
                                 overwrite=True,
                                 optimizer=optimizer,
                                 clip_grad=True,
                                 rank=gpu, train_function=training.train, gpus=opt.gpus, n_view=opt.views, model_name=opt.model)

if __name__ == "__main__":
    opt = config.config_parser()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)
