import math
import numbers
import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from rmi import RMILoss

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, num_queries=100):
        super(ContrastiveLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.num_queries = num_queries
        self.labels = torch.eye(num_queries)    #
        self.cos_sim = nn.CosineSimilarity()
        self.mi_score = RMILoss(with_logits=False)
        self.contr_sim = SimCLR_Loss(num_queries, 0.5)

    def random_idx(self):
        idx_list = []
        for _ in range(2):
            idx = random.randint(0, self.num_queries-1)
            while idx in idx_list:
                idx = random.randint(0, self.num_queries-1)
            idx_list.append(idx)
        idx_list = np.array(idx_list)
        return idx_list

    def forward(self, query1, query2, init_query):
        B = query1.shape[0]
        labels = self.labels.unsqueeze(0).repeat(B, 1, 1)   # [B, Q, Q]
        labels = labels.to(query1.device)

        # Matching
        # query_sim = query1 @ query2.permute(0, 2, 1)    #[B, Q1, Q2]
        # loss1 = self.mse(query_sim, labels)
        # loss1 /= self.num_queries**2
        loss1 = torch.stack([self.contr_sim(query1[b], query2[b]) for b in range(B)]).mean()

        # Entropy
        # probabilities = F.softmax(init_query, dim=-1)
        # entropy = -torch.sum(probabilities* torch.log(probabilities + 1e-10), dim=-1)
        # loss2 = entropy.mean()
        sel_idx = self.random_idx()
        probabilities = F.softmax(init_query, dim=-1)
        sel_query = probabilities[:, sel_idx] # [B, 2, 256]
        sel_query = sel_query.unsqueeze(0)
        loss2 = self.mi_score(sel_query[:, :, 0:1], sel_query[:, :, 1:2])
        return loss1 + loss2

def image_loss(model_out, gt, mask=None):
    gt_rgb = gt['rgb']
    gt_rgb[torch.isnan(gt_rgb)] = 0.0
    rgb = model_out['rgb']
    rgb[torch.isnan(rgb)] = 0.0
    loss = torch.abs(gt_rgb - rgb).mean()
    return loss

class LFLoss():
    def __init__(self, l2_weight=1e-3, lpips=False, depth=False, contra=False):
        self.l2_weight = l2_weight
        self.contra = contra
        self.lpips = lpips
        self.depth = depth

        import lpips
        loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
        self.loss_fn_alex = loss_fn_alex.cuda()
        self.smooth = GaussianSmoothing(1, 15, 7).cuda()

        self.avg = nn.AdaptiveAvgPool2d((2, 2))
        self.upsample = nn.Upsample((32, 32), mode='bilinear')

    def __call__(self, model_out, gt, model=None, val=False):
        loss_dict = {}
        loss_dict['img_loss'] = image_loss(model_out, gt)

        if self.lpips:
            gt_rgb = gt['rgb']                      # [B, num_ctxt_views, H, W, 3]
            mask = gt['mask']
            pred_rgb = model_out['rgb']
            valid_mask = model_out['valid_mask']
            offset = 32
            gt_rgb = gt_rgb.reshape((-1, offset, offset, 3)).permute(0, 3, 1, 2)
            pred_rgb = pred_rgb.reshape((-1, offset, offset, 3)).permute(0, 3, 1, 2)
            
            if mask.size(0) == gt_rgb.size(0):
                gt_rgb = gt_rgb * mask[:, None, None, None]
                pred_rgb = pred_rgb * mask[:, None, None, None]

            lpips_loss = self.loss_fn_alex(gt_rgb, pred_rgb)

            # 0.2 for realestate
            loss_dict['lpips_loss'] = 0.1 * lpips_loss

        if self.depth and (not val):
            depth_ray = model_out['depth_ray'][..., 0]
            depth_ray = depth_ray.reshape((-1, 1, 32, 32))

            depth_mean = depth_ray.mean(dim=-1).mean(dim=-1)[:, None, None]
            depth_dist = self.l2_weight * torch.pow(depth_ray - depth_mean, 2).mean(dim=-1).mean(dim=-1).mean(dim=-1)

            mask = gt['mask']
            depth_loss = depth_dist * mask[:, None]
            loss_dict['depth_loss'] = depth_loss.mean()
        
        if self.contra and (not val):
            contra_losses = model_out['contra_losses']
            loss_dict['contra_loss'] = contra_losses.mean()
            
        return loss_dict, {}