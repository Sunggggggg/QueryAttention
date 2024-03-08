import os
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from models import CrossAttentionRenderer
from dataset.realestate10k_dataio import RealEstate10k
import configargparse
from utils import util
from torch.utils.tensorboard import SummaryWriter

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

def config_parser():
    # python queries_visualization --logging_root realestate_query/64_2_512
    parser = configargparse.ArgumentParser()

    parser.add_argument('--logging_root', type=str)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    return parser.parse_args()

if __name__ == "__main__" :
    args = config_parser()

    #Load dataset
    val_dataset = RealEstate10k(img_root="/home/dev4/data/SKY/datasets/data_download/realestate/test",
                                pose_root="/home/dev4/data/SKY/datasets/poses/realestate/test.mat",
                                num_ctxt_views=2, num_query_views=1, augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, 
                            drop_last=True, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)
    #Model
    model = CrossAttentionRenderer(no_multiview=False, 
                                     no_sample=False, 
                                     no_latent_concat=False, 
                                     no_high_freq=False, 
                                     model='query', n_view=2)
    
    
    weight_path = os.path.join(args.logging_root, 'checkpoints', 'model_current.pth')
    weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights['model'], strict=False)
    model.cuda().eval()
    # 
    backbone = model.encoder.backbone
    query_feat = model.encoder.query_feat
    query_embed = model.encoder.query_embed

    query_featmap_path = os.path.join(args.logging_root, 'QueryAttention')
    os.makedirs(query_featmap_path, exist_ok=True)
    writer = SummaryWriter(query_featmap_path, flush_secs=10)

    total_iter = 0
    with torch.no_grad():
        for val_i, (model_input, gt) in enumerate(val_dataloader):
            model_input = util.dict_to_gpu(model_input)
            gt = util.dict_to_gpu(gt)

            query_images = model_input['query']['rgb']
            query_images = query_images.permute(0, 3, 1, 2)
            writer.add_image("GT",
                             torchvision.utils.make_grid(query_images, scale_each=False, normalize=True).cpu().numpy(), total_iter)


            z, _ = model.get_z(model_input)     # [B, dim1, H, W], [B, dim2, H, W], 

            for i, feat in enumerate(z) :
                writer.add_image(f"Attention Maps{i}", 
                                 torchvision.utils.make_grid(feat, scale_each=False, normalize=True).cpu().numpy(), total_iter)
            
            
            break
