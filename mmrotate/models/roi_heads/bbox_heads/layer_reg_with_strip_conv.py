import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_crosswise_conv_module import SpatialCrosswiseConvModule


class LayerRegWithStripConv(nn.Module):
    """LayerReg with Spatial Crosswise Convolution Module
    
    x has shape (B, N, C)

               B, N, C
    reshape -> B, C, A, A
    conv3x3 -> B, C, (A - 2), (A - 2)
    conv3x3 -> B, C, (A - 4), (A - 4)
    conv3x3 -> B, C, (A - 6), (A - 6)
    strip_conv_module -> B, C, (A - 6), (A - 6)  # NEW: Spatial Crosswise Conv Module
    reshape -> B, C
    fc      -> B, out_channels
    """
    def __init__(self, in_channels=256, out_channels=2, num_convs=0, feat_size=7):
        super().__init__()
        self.num_convs = num_convs
        self.feat_size = feat_size

        if self.num_convs > 0:
            self.norms = nn.ModuleList()
            self.convs = nn.ModuleList()
            self.relu = nn.ReLU(True)
            for i in range(self.num_convs):
                self.norms.append(nn.LayerNorm([in_channels, feat_size - i * 2, feat_size - i * 2]))
                self.convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0))
            self.last_feat_area = (feat_size - num_convs * 2) ** 2
            
            # Add Spatial Crosswise Convolution Module after conv layers
            self.strip_conv_module = SpatialCrosswiseConvModule(in_channels)
        else:
            self.strip_conv_module = None

        self.norm_reg = nn.LayerNorm(in_channels)
        self.fc_reg = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        if self.num_convs > 0:
            B, N, C = x.shape
            x = x.transpose(-2, -1).reshape(B, C, self.feat_size, self.feat_size)
            
            # Original conv layers
            for i in range(self.num_convs):
                norm = self.norms[i]
                conv = self.convs[i]
                x = self.relu(conv(norm(x)))
            
            # NEW: Apply Spatial Crosswise Convolution Module
            if self.strip_conv_module is not None:
                x = self.strip_conv_module(x)
            
            # Reshape back to (B, N, C) format
            x = x.reshape(B, C, self.last_feat_area).transpose(-2, -1)
        else:
            # If no conv layers, still apply strip conv module if input is suitable
            B, N, C = x.shape
            feat_size = int(N ** 0.5)  # Assume square feature map
            if feat_size ** 2 == N and self.strip_conv_module is not None:
                x = x.transpose(-2, -1).reshape(B, C, feat_size, feat_size)
                x = self.strip_conv_module(x)
                x = x.reshape(B, C, N).transpose(-2, -1)

        bbox_pr = self.fc_reg(self.norm_reg(x.mean(dim=1)))
        return bbox_pr