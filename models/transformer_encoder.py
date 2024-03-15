import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from loss_functions import ContrastiveLoss
from .backbone import Backbone
from .attention import *

def QueryAggregation(queries1, queries2):
    """
    queries1, queries2 : [B, Q, e]
    """
    matching_score = torch.matmul(queries1, queries2.transpose(1, 2))
    matching_score1 = matching_score.softmax(dim=1)
    matching_score2 = matching_score.softmax(dim=2)

    expect_queries = \
        (torch.matmul(matching_score1.transpose(1, 2), queries1) + torch.matmul(matching_score2, queries2))/2
    
    return expect_queries

class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups=1
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        super(FeatureFusionBlock_custom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups=1
        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output

class MultiviewEncoder(nn.Module):
    def __init__(self, name='resnet50', num_feat_levels=3, 
                 num_queries=16, hidden_dim=256, dim_feedforward=2048, nheads=8, depth=9) :
        super(MultiviewEncoder, self).__init__()
        self.num_feat_levels = num_feat_levels
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Backbone
        self.backbone = Backbone(name=name, pretrained=True, freeze=True, num_feat_levels=num_feat_levels, hidden_dim=hidden_dim)

        # Learnable query
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Camera pose embedding
        self.cam_pose_embed = nn.Linear(4*4, hidden_dim)

        # level embedding
        self.level_embed = nn.Embedding(num_feat_levels, hidden_dim)    # [3, 256]

        # Block
        self.query_activation1 = nn.ModuleList()
        self.query_activation2 = nn.ModuleList()
        self.self_attention_layers1 = nn.ModuleList()
        self.self_attention_layers2 = nn.ModuleList()
        self.cross_attention_layers1 = nn.ModuleList()
        self.cross_attention_layers2 = nn.ModuleList()
        self.ffn_layers1 = nn.ModuleList()
        self.ffn_layers2 = nn.ModuleList()
        self.ffn_layers3 = nn.ModuleList()
        self.ffn_layers4 = nn.ModuleList()


        # 1. Query Activation
        # 2. Self Attention
        # 3. Feed Forward
        # 4. Cross Attention
        # 5. Feed Forward
        # 6. 
        for _ in range(depth) :
            # 1. Query Activation
            self.query_activation1.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0))
            self.query_activation2.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0))
            # 2. Self Attention
            self.self_attention_layers1.append(SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0))
            self.self_attention_layers2.append(SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0))
            # 3. Feed Forward
            self.ffn_layers1.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0))
            self.ffn_layers2.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0))
            # 4. Cross Attention
            self.cross_attention_layers1.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0))
            self.cross_attention_layers2.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0))
            # 5. Feed Forward
            self.ffn_layers3.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0))
            self.ffn_layers4.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0))

        self.refinenet1 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet2 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet3 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)

        # Loss func
        self.loss_func = ContrastiveLoss(self.num_queries)

    def forward(self, x, rel_transform, nviews=2):
        """
        x               : [2B, 3, H, W]
        rel_transform   : [2B, 16]
        n_view          : 2
        """
        s = x.shape
        x = x.reshape(s[0]//nviews, nviews, *s[1:])
        img1, img2 = x[:, 0], x[:, 1]       # [B, 3, H, W]

        # Query embedding
        B = x.shape[0]
        query = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, e]
        query1, query2 = query, query

        # Camera pose embedding
        pose_embed = self.cam_pose_embed(rel_transform)                     # [2B, hidden_dim]
        pose_embed = pose_embed.reshape(s[0]//nviews, nviews, -1)

        # Feature map
        keypoint_maps =[]
        feats1, feats2 = self.backbone(img1), self.backbone(img2)   # [layer3, layer2, layer1]
        for i in range(self.depth) :
            level_index = i % self.num_feat_levels                      # [0, 1, 2]
            feat1, feat2 = feats1[level_index], feats2[level_index]     # [B, hidden_dim, H, W]
            B, _, h, w = feat1.shape 

            feat1 = feat1.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]
            feat2 = feat2.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]

            #
            pose_embed1 = pose_embed[:, 0].unsqueeze(1).repeat(1, h*w, 1)    # [B, hw, e]
            pose_embed2 = pose_embed[:, 1].unsqueeze(1).repeat(1, h*w, 1) 
            # 
            level_embed = self.level_embed.weight[level_index].unsqueeze(0).unsqueeze(0).repeat(B, h*w, 1)  # [B, hw, e]
           
            query1 = self.query_activation1[i](query1, feat1, cam_pos=pose_embed1+level_embed, query_pos=query_embed)
            query2 = self.query_activation2[i](query2, feat2, cam_pos=pose_embed2+level_embed, query_pos=query_embed)
            # 
            query1 = self.self_attention_layers1[i](query1, query_pos=query_embed)
            query2 = self.self_attention_layers2[i](query2, query_pos=query_embed)
            # 
            query1 = self.ffn_layers1[i](query1)
            query2 = self.ffn_layers2[i](query2)
            #
            _query1, _query2 = query1, query2
            query1 = self.cross_attention_layers1[i](query1, _query2, cam_pos=query_embed, query_pos=query_embed)
            query2 = self.cross_attention_layers2[i](query2, _query1, cam_pos=query_embed, query_pos=query_embed)

            query1 = self.ffn_layers3[i](query1)
            query2 = self.ffn_layers4[i](query2)
            #

            contra_loss = self.loss_func(query1, query2)

        query = (query1 + query2) / 2
        # Make pixel align
        for i in range(self.num_feat_levels) :
            feat1, feat2 = feats1[i], feats2[i]
            B, _, h, w = feat1.shape 

            feat1 = feat1.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]
            feat2 = feat2.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]
    
            # 
            keypoint_map1 = torch.matmul(query, feat1.transpose(1,2)).reshape(B, self.num_queries, h, w)      # [B, Q, e]*[B, e, hw] = [B, Q, h, w]
            keypoint_map2 = torch.matmul(query, feat2.transpose(1,2)).reshape(B, self.num_queries, h, w)

            keypoint_map = torch.stack([keypoint_map1, keypoint_map2], dim=1)                 
            keypoint_map = torch.flatten(keypoint_map, 0, 1)                # [2B, Q, H, W]

            keypoint_maps.append(keypoint_map)
        
        # 
        assert self.num_feat_levels == 3    # Fix 3 scale
        path_3 = self.refinenet3(keypoint_maps[2])
        path_2 = self.refinenet2(path_3, keypoint_maps[1])
        path_1 = self.refinenet1(path_2, keypoint_maps[0])

        return [path_2, path_1], contra_loss
    
if __name__ == '__main__' :
    x = torch.rand((4, 3, 256, 256))
    y = torch.rand((4, 16))
    print(y.device)
    model = MultiviewEncoder()
    model(x, y)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))