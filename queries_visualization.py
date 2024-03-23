import os
import random
import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from models import CrossAttentionRenderer
from dataset.realestate10k_dataio import RealEstate10k
import configargparse
from utils import util
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
                                     model='query', 
                                     n_view=2,
                                     num_queries=args.num_queries
                                     )
    
    backbone = model.encoder.backbone

    weight_path = os.path.join(args.logging_root, 'checkpoints', 'model_current.pth')
    weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights['model'], strict=False)
    model.cuda().eval()

    # 
    query_featmap_path = os.path.join(args.logging_root, 'QueryAttention')
    os.makedirs(query_featmap_path, exist_ok=True)
    writer = SummaryWriter(query_featmap_path, flush_secs=10)

    total_iter = 0
    with torch.no_grad():
        for val_i, (model_input, gt) in enumerate(val_dataloader):
            model_input = util.dict_to_gpu(model_input)
            gt = util.dict_to_gpu(gt)

            rgb_full = model_input['query']['rgb']
            uv_full = model_input['query']['uv']

            nrays = uv_full.size(2)
            z, _ = model.get_z(model_input)
         
            rgb_chunks = torch.chunk(rgb_full, 9, dim=2)
            uv_chunks = torch.chunk(uv_full, 9, dim=2)

            model_outputs = []

            for rgb_chunk, uv_chunk in zip(rgb_chunks, uv_chunks):
                model_input['query']['rgb'] = rgb_chunk
                model_input['query']['uv'] = uv_chunk

                model_output = model(model_input, z=z, val=True)

                model_outputs.append(model_output)

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
            rgb = ((rgb + 1) * 0.5).detach() * valid_mask + 0.5 * (1 - valid_mask) * torch.ones_like(rgb)
            rgb = torch.clamp(rgb, -1, 1)
            
            writer.add_image("Prediction", torchvision.utils.make_grid(rgb[None, ...].permute(0, 3, 1, 2), scale_each=False)
                              , total_iter)

            # 
            query_images = rgb_full      #[B, 1, HW, 3]
            query_images = query_images.view(1, 256, 256, 3)
            query_images = query_images.permute(0, 3, 1, 2)
            writer.add_image("Queryview",
                             torchvision.utils.make_grid(query_images, scale_each=False, normalize=True).cpu().numpy(), total_iter)

            context_images = util.flatten_first_two(model_input['context']['rgb'])# [2B, H, W, 3]
            _context_images = context_images.permute(0, 3, 1, 2)                   # [2B, 3, H, W]
            writer.add_image("Contextviews",
                             torchvision.utils.make_grid(_context_images, scale_each=False, normalize=True).cpu().numpy(), total_iter)
            
            # 
            def norm(img):
                low, high = img.min(), img.max()
                img.clamp_(min=low, max=high)
                img.sub_(low).div_(max(high - low, 1e-5))
                return img
            
            context_images = torch.stack([norm(context) for context in context_images])

            feat1, feat2 = backbone(_context_images)
            feat1, feat2 = feat1[0], feat2[0]
            for k in range(feat1.shape[1]):
                writer.add_image(f"feature Maps{k}", 
                    torchvision.utils.make_grid(torch.stack([feat1[:, k:k+1], feat2[:, k:k+1]]), scale_each=False, normalize=True).cpu().numpy(), total_iter)


            high_feat = z[1]
            for k in range(high_feat.shape[1]) :
                query1, query2 = model.encoder.query1, model.encoder.query2 # [1, 100, 256]
                tsne = TSNE(n_components=2, random_state=1)
                query1_tsne = tsne.fit_transform(query1)
                query2_tsne = tsne.fit_transform(query2)
                color = np.arange(query2.shape[1])
                plt.scatter(query1_tsne[0, :, 0], query1_tsne[0, :, 1], c=color)
                writer.add_figure(f'query1_embedding{k}', plt.gcf(), total_iter)
                plt.scatter(query2_tsne[0, :, 0], query2_tsne[0, :, 1], c=color)
                writer.add_figure(f'query2_embedding{k}', plt.gcf(), total_iter)

                featmaps = high_feat[:, k:k+1]                      # [2, 1, H, W]
                mask = featmaps.permute(0, 2, 3, 1).cpu().numpy()   # [2, H, W, 1]
                mask1, mask2 = mask[0], mask[1]                     
                mask1 = mask1 / mask1.max()
                mask2 = mask2 / mask2.max()

                # mask1 = np.where(mask1 >= 0.5, np.float32(1.0), np.float32(0.0))
                # mask2 = np.where(mask2 >= 0.5, np.float32(1.0), np.float32(0.0))

                cam1 = mask1 + np.float32(context_images[0].cpu().numpy())
                cam2 = mask2 + np.float32(context_images[1].cpu().numpy())
                cam1 = cam1 / np.max(cam1)
                cam2 = cam2 / np.max(cam2)

                cam1 = np.uint8(255 * cam1)
                cam2 = np.uint8(255 * cam2)

                cam1 = cv2.applyColorMap(cam1, cv2.COLORMAP_JET)  # [H, W, 3]
                cam2 = cv2.applyColorMap(cam2, cv2.COLORMAP_JET)  # [H, W, 3] 
                cam = np.stack([cam1, cam2], axis=0)               # [2, H, W, 3]

                cam = cam.transpose(0, -1, 1, 2)
                writer.add_image(f"Attention Maps{k}", 
                                torchvision.utils.make_grid(torch.tensor(cam), scale_each=False), total_iter)
        
            
            break
