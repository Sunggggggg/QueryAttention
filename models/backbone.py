import torchvision
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, name='resnet50', pretrained=True, freeze=True, num_feat_levels=4, hidden_dim=256) :
        super(Backbone, self).__init__()
        self.name = name
        self.num_feat_levels = num_feat_levels
        # 
        if name == 'resnet50' and pretrained:
            #weights = torchvision.models.ResNet50_Weights
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            feat_dim = [2**8, 2**9, 2**10]
        
        elif name == 'swin_b' and pretrained :
            #weights = torchvision.models.Swin_B_Weights
            backbone = torchvision.models.swin_b(pretrained=pretrained)
            feat_dim = [2**7, 2**8, 2**9]
        
        if freeze :
            for name, param in backbone.named_parameters():
                param.requires_grad_(False)

        self.backbone = backbone

        # Same feature dim
        self.branch = nn.Module()
        self.branch.layer1 = nn.Conv2d(feat_dim[0], hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.branch.layer2 = nn.Conv2d(feat_dim[1], hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.branch.layer3 = nn.Conv2d(feat_dim[2], hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=1)

    def forward(self, x):
        if self.name == 'resnet50':
            #
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)

            # 
            layer1 = self.backbone.layer1(x)        # 256
            layer2 = self.backbone.layer2(layer1)   # 512
            layer3 = self.backbone.layer3(layer2)   # 1024

        elif self.name == 'swin_b' :
            layer1 = self.backbone.features[0](x)
            layer1 = self.backbone.features[1](layer1)  # 128

            layer2 = self.backbone.features[2](layer1)
            layer2 = self.backbone.features[3](layer2)  # 256

            layer3 = self.backbone.features[4](layer2)
            layer3 = self.backbone.features[5](layer3)  # 512

            layer1 = layer1.permute(0, 3, 1, 2)
            layer2 = layer2.permute(0, 3, 1, 2)
            layer3 = layer3.permute(0, 3, 1, 2)

        layer1 = self.branch.layer1(layer1)   # hidden_dim
        layer2 = self.branch.layer2(layer2)   # hidden_dim
        layer3 = self.branch.layer3(layer3)   # hidden_dim
        
        return [layer3, layer2, layer1][::-1]