import torch
import torch.nn as nn
import torchvision
from models import MultiScaleQueryTransformerDecoder


if __name__ == "__main__" :
    encoder = MultiScaleQueryTransformerDecoder(num_layers=3, num_queries=100, hidden_dim=256, nheads=8, depth=6)
    
    weight_path = 'realestate_query/64_3_512/checkpoints/model_current.pth'
    weights = torch.load(weight_path)
    encoder.load_state_dict(weight_path['model'], strict=False)

    # 
    backbone = encoder.backbone
    query_feat = encoder.query_feat
    query_embed = encoder.query_embed

    img1, img2 = torch.rand((2, 3, 256, 256))
    feats1, feats2 = backbone(img1), backbone(img2)
    # Mat mul 해서 어디에 Heat되는지 확인