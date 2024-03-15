import torch
import torch.nn as nn
from backbone import Backbone
from .attention import *

class MultiviewEncoder(nn.Module):
    def __init__(self, name='resnet50', num_feat_levels=3, num_queries=100, hidden_dim=256, dim_feedforward=2048, nheads=8, num_depth=6) :
        self.num_feat_levels = num_feat_levels
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_depth = num_depth

        # Backbone
        self.backbone = Backbone(name=name, pretrained=True, freeze=True, num_feat_levels=num_feat_levels, hidden_dim=hidden_dim)

        # Learnable query
        self.query_feat = nn.Parameter(torch.rand(num_queries, hidden_dim), requires_grad=True) 
        self.query_embed = nn.Parameter(torch.rand(num_queries, hidden_dim), requires_grad=True) 

        # Camera pose embedding
        self.cam_pose_embed = nn.Linear(4*4, hidden_dim)

        # Level embedding
        self.level_embed = nn.Parameter(torch.rand(num_feat_levels, hidden_dim), requires_grad=True)    # [3, 256]

        # Block
        self.query_activation = nn.ModuleList()
        self.attention_layers_q = nn.ModuleList()

        for level in range(num_feat_levels):
            self.query_activation.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0))

            for depth in range(num_depth) :    
                self.attention_layers.append(AttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0))


    def forward(self, x, rel_transform, nviews=2):
        # 
        s = x.shape
        x = x.reshape(s[0]//nviews, nviews, *s[1:])
        img1, img2 = x[:, 0], x[:, 1]       # [B, 3, H, W]

        # Camera pose embedding
        pose_embed = self.cam_pose_embed(rel_transform)                     # [2B, hidden_dim]
        pose_embed = pose_embed.reshape(s[0]//nviews, nviews, -1)           # [B, 2, e]

        # Learnable query init
        B = x.shape[0]
        init_query = self.query_feat.unsqueeze(0).repeat(B, 1, 1)           # [B, Q, e]
        query_embed = self.query_embed.unsqueeze(0).repeat(B, 1, 1)         # [B, Q, e]

        # Multiscale feature map
        feats1, feats2 = self.backbone(img1), self.backbone(img2)           # [layer3, layer2, layer1]
        for level in range(self.num_feat_levels):
            feat1, feat2 = feats1[level], feats2[level]     # [B, e, h, w]
            
            feat1 = feat1.flatten(-2).permute(0, 2, 1)
            feat2 = feat2.flatten(-2).permute(0, 2, 1)

            pose_embed1 = pose_embed[:, 0]                  # [e]
            pose_embed1 = pose_embed1.view(1, 1, -1)          # [B(1), hw(1), e]

            pose_embed2 = pose_embed[:, 0]                  # [e]
            pose_embed2 = pose_embed2.view(1, 1, -1)        # [B, hw, e]

            query1 = self.query_activation[level](init_query, feat1, query_pos=query_embed, key_pos=pose_embed1)
            query2 = self.query_activation[level](init_query, feat2, query_pos=query_embed, key_pos=pose_embed2)
            cost_volume = query1 @ query2.permute(0, 2, 1)  # [B, Q1, Q2]

            cost_feat1 = torch.cat([cost_volume, query1], dim=-1)                    # [B, Q1, (Q2+e)]
            cost_feat2 = torch.cat([cost_volume.permute(0, 2, 1), query2], dim=-1)   # [B, Q2, (Q1+e)]

            for depth in range(self.num_depth):
                idx = depth + depth*level
                cost_volume1 = self.attention_layers[idx](cost_feat1, cost_feat1, cost_feat1)
                cost_volume2 = self.attention_layers[idx](cost_feat2, cost_feat2, cost_feat2)

                query1 = self.attention_layers(cost_feat1, cost_feat1, query1)
                query2 = self.attention_layers(cost_feat2, cost_feat2, query2)

                query_embed1 = query_embed + pose_embed1
                query_embed2 = query_embed + pose_embed2
