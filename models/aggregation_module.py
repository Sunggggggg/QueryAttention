import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import models.resnet as resnet
from operator import add
from functools import reduce
from loss_functions import ContrastiveLoss
from .backbone import Backbone

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

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [B, C, H, W]
        return pos.flatten(-2)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class Attention(nn.Module):
    """
    Full Attention : No Multi head
    """
    def __init__(self, d_model, query_dim=None, key_dim=None, value_dim = None, nhead=1, dropout=0.0, activation="relu"):
        super(Attention, self).__init__()
        self.num_heads = nhead
        head_dim = d_model // nhead
        self.scale = head_dim ** -0.5

        if query_dim is not None :
            self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        else :
            self.q_proj = nn.Linear(d_model, d_model, bias=False)

        if key_dim is not None :
            self.k_proj = nn.Linear(key_dim, key_dim, bias=False)
        else :
            self.k_proj = nn.Linear(d_model, d_model, bias=False)

        if value_dim is not None :
            self.v_proj = nn.Linear(value_dim, value_dim, bias=False)
        else :
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value, query_pos=None, key_pos=None):
        """
        query, key, value : [B, Q, e]
        """
        _query = query

        query = self.with_pos_embed(query, query_pos)
        key = self.with_pos_embed(key, key_pos)

        # query = self.q_proj(query).view(B, -1, self.num_heads, C//self.num_heads) 
        # key = self.k_proj(key).view(B, -1, self.num_heads, C//self.num_heads)      
        # value = self.v_proj(value).view(B, -1, self.num_heads, C//self.num_heads)
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        attn = (query @ key.transpose(-2, -1)) * self.scale # [B, Q1, e] @ [B, e, Q1] 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value            # [B, Q1, Q2] @ [B, Q2, e]  = [B, Q1, e]
        return x

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super(FFNLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
    
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, queries):
        queries2 = self.linear2(self.dropout(self.activation(self.linear1(queries))))
        queries = queries + self.dropout(queries2)
        queries = self.norm(queries)
        return queries

class FeatureExtractionHyperPixel(nn.Module):
    def __init__(self, hyperpixel_ids, feature_size, freeze=True):
        super(FeatureExtractionHyperPixel, self).__init__()
        self.backbone = resnet.resnet101(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids
    
    def forward(self, img):
        """
        [B, ]
        """
        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)
            print(feats[idx].shape)

        return feats[:3][::-1]

class MultiviewEncoder(nn.Module):
    def __init__(self, name='resnet50', num_feat_levels=3, num_queries=32, hidden_dim=256, dim_feedforward=2048, nheads=1, num_depth=12) :
        super(MultiviewEncoder, self).__init__()
        self.num_feat_levels = num_feat_levels
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_depth = num_depth

        # Backbone
        self.backbone = Backbone(name=name, pretrained=True, freeze=True, num_feat_levels=num_feat_levels, hidden_dim=hidden_dim)
        #self.backbone = FeatureExtractionHyperPixel([0,8,20,21,26,28,29,30], hidden_dim, True)
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # Learnable query
        self.query_feat = nn.Parameter(torch.rand(num_queries, hidden_dim), requires_grad=True) 
        self.query_embed_x = nn.Parameter(torch.rand(num_queries, hidden_dim), requires_grad=True) 

        # 
        self.query_embed_y1 = nn.Parameter(torch.rand(num_queries, hidden_dim), requires_grad=True)
        self.query_embed_y2 = nn.Parameter(torch.rand(num_queries, hidden_dim), requires_grad=True)

        # Block
        self.query_activation1 = nn.ModuleList()
        self.query_activation2 = nn.ModuleList()
        self.self_attention_layers_query = nn.ModuleList()
        self.self_attention_layers_cost_vol = nn.ModuleList()
        self.ffn_layers1 = nn.ModuleList()
        self.ffn_layers2 = nn.ModuleList()
        self.cross_attention_layers_query = nn.ModuleList()
        self.cross_attention_layers_cost_vol = nn.ModuleList()

        for _ in range(num_depth) :
            self.query_activation1.append(Attention(d_model=hidden_dim, nhead=nheads, dropout=0.0))
            self.query_activation2.append(Attention(d_model=hidden_dim, nhead=nheads, dropout=0.0))
            self.self_attention_layers_query.append(
                Attention(d_model=hidden_dim, query_dim=hidden_dim+num_queries, key_dim=hidden_dim+num_queries, value_dim=hidden_dim, nhead=nheads, dropout=0.0)
                )
            self.self_attention_layers_cost_vol.append(
                Attention(d_model=hidden_dim, query_dim=hidden_dim+num_queries, key_dim=hidden_dim+num_queries, value_dim=num_queries, nhead=nheads, dropout=0.0)
                )
            self.cross_attention_layers_query.append(
                Attention(d_model=hidden_dim, query_dim=hidden_dim+num_queries, key_dim=hidden_dim+num_queries, value_dim=hidden_dim, nhead=nheads, dropout=0.0)
                )
            self.cross_attention_layers_cost_vol.append(
                Attention(d_model=hidden_dim, query_dim=hidden_dim+num_queries, key_dim=hidden_dim+num_queries, value_dim=num_queries, nhead=nheads, dropout=0.0)
                )
            self.ffn_layers1.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0))
            self.ffn_layers2.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0))

        self.loss_func = ContrastiveLoss(self.num_queries)

        self.refinenet1 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet2 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)
        self.refinenet3 = FeatureFusionBlock_custom(num_queries, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True)


        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.norm5 = nn.LayerNorm(hidden_dim)
        self.norm6 = nn.LayerNorm(hidden_dim)
        self.norm7 = nn.LayerNorm(hidden_dim)
        self.norm8 = nn.LayerNorm(hidden_dim)
        self.norm9 = nn.LayerNorm(hidden_dim)
        self.norm10 = nn.LayerNorm(hidden_dim)

    def forward(self, x, rel_transform, nviews=2):
        # 
        s = x.shape
        x = x.reshape(s[0]//nviews, nviews, *s[1:])
        img1, img2 = x[:, 0], x[:, 1]       # [B, 3, H, W]

        # Learnable query init
        B = x.shape[0]
        init_query = self.query_feat.unsqueeze(0).repeat(B, 1, 1)                   # [B, Q, e]
        query_embed_x = self.query_embed_x.unsqueeze(0).repeat(B, 1, 1)             # [B, Q, e]
        query1, query2 = init_query, init_query

        # Multiscale feature map
        queries1, queries2, contra_losses = [], [], []
        feats1, feats2 = self.backbone(img1), self.backbone(img2)           # [layer3, layer2, layer1]
        for level in range(self.num_feat_levels):
            # Feature map
            feat1, feat2 = feats1[level], feats2[level]     # [B, e, h, w]
            # Key pos_embed
            key_pos1 = self.pe_layer(feat1)
            key_pos2 = self.pe_layer(feat2)                 # [B, e, hw]
            key_pos1 = key_pos1.permute(0, 2, 1)            # [B, hw, e]
            key_pos2 = key_pos2.permute(0, 2, 1)

            feat1 = feat1.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]
            feat2 = feat2.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]
            
            # Query pos_embed
            query_pos1 = query_embed_x + self.query_embed_y1.unsqueeze(0).repeat(B, 1, 1)
            query_pos2 = query_embed_x + self.query_embed_y2.unsqueeze(0).repeat(B, 1, 1) # [B, Q, e]

            for depth in range(self.num_depth):
                # Query Activation
                query1 = query1 + self.query_activation1[depth](query1, query1, query1, query_pos=query_embed_x)
                query2 = query2 + self.query_activation1[depth](query2, query2, query2, query_pos=query_embed_x)
                query1 = self.norm1(query1)
                query2 = self.norm2(query1)

                query1 = query1 + self.query_activation2[depth](query1, feat1, feat1, query_pos=query_pos1, key_pos=key_pos1)
                query2 = query2 + self.query_activation2[depth](query2, feat2, feat2, query_pos=query_pos2, key_pos=key_pos2)
                query1 = self.norm3(query1)
                query2 = self.norm4(query1)
                cost_volume = query1 @ query2.permute(0, 2, 1)  # [B, Q1, Q2]

                # Intra Aggreagation
                cost_feat1 = torch.cat([cost_volume, query1], dim=-1)                    # [B, Q1, (Q2+e)]
                cost_feat2 = torch.cat([cost_volume.permute(0, 2, 1), query2], dim=-1)   # [B, Q2, (Q1+e)]

                _query1, _query2 = query1, query2
                query1 = query1 + self.self_attention_layers_query[depth](cost_feat1, cost_feat1, query1)
                query1 = self.norm5(query1)
                query1 = self.ffn_layers1[depth](query1)
                cost_volume1 = self.self_attention_layers_cost_vol[depth](cost_feat1, cost_feat1, cost_volume) # [B, Q1, Q2]
                query1 = query1 + cost_volume1.softmax(dim=1) @ _query2      # [B, Q1, Q2] [B, Q2, e]
                query1 = self.norm6(query1)

                query2 = query2 + self.self_attention_layers_query[depth](cost_feat2, cost_feat2, query2)   # [B, Q2, e]
                query2 = self.norm7(query2)
                query2 = self.ffn_layers1[depth](query2)
                cost_volume2 = self.self_attention_layers_cost_vol[depth](cost_feat2, cost_feat2, cost_volume.permute(0, 2, 1)) # [B, Q2, Q1]
                query2 = query2 + cost_volume2.softmax(dim=1) @ _query1     # [B, Q2, Q1] [B, Q1, e]
                query2 = self.norm8(query2)
                
                # Inter Aggregtion
                cost_feat1 = torch.cat([cost_volume, query1], dim=-1)                    # [B, Q1, (Q2+e)]
                cost_feat2 = torch.cat([cost_volume.permute(0, 2, 1), query2], dim=-1)   # [B, Q2, (Q1+e)]

                _query1, _query2 = query1, query2
                query1 = self.cross_attention_layers_query[depth](cost_feat1, cost_feat2, _query2)  # []
                query1 = self.norm9(query1)
                query1 = self.ffn_layers2[depth](query1)

                query2 = self.cross_attention_layers_query[depth](cost_feat2, cost_feat1, _query1)   # [B, Q2, e]
                query2 = self.norm10(query2)
                query2 = self.ffn_layers2[depth](query2)

            queries1.append(query1)
            queries2.append(query2)
            contra_losses.append(self.loss_func(query1, query2))

        # Make pixel align
        keypoint_maps =[]
        for i in range(self.num_feat_levels) :
            feat1, feat2 = feats1[i], feats2[i]
            query1, query2 = queries1[i], queries2[i]
            B, _, h, w = feat1.shape 

            feat1 = feat1.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]
            feat2 = feat2.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)   # [B, hw, e]
    
            # 
            keypoint_map1 = torch.matmul(query1, feat1.transpose(1,2)).reshape(B, self.num_queries, h, w)      # [B, Q, e]*[B, e, hw] = [B, Q, h, w]
            keypoint_map2 = torch.matmul(query2, feat2.transpose(1,2)).reshape(B, self.num_queries, h, w)

            keypoint_map = torch.stack([keypoint_map1, keypoint_map2], dim=1)                 
            keypoint_map = torch.flatten(keypoint_map, 0, 1)                # [2B, Q, H, W]

            keypoint_maps.append(keypoint_map)

        assert self.num_feat_levels == 3    # Fix 3 scale
        path_3 = self.refinenet3(keypoint_maps[2])
        path_2 = self.refinenet2(path_3, keypoint_maps[1])
        path_1 = self.refinenet1(path_2, keypoint_maps[0])

        return [path_2, path_1], contra_losses
    
if __name__ == '__main__' :
    x = torch.rand((4, 3, 256, 256))
    y = torch.rand((4, 16))
    print(y.device)
    model = MultiviewEncoder()
    model(x, y)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))