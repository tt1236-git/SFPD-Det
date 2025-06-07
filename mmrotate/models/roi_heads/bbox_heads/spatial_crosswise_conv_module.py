import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCrosswiseConvModule(nn.Module):
    """Spatial Crosswise Convolution Module
    
    This module implements the strip convolution structure described in the paper.
    The module structure is:
    1. Input -> 3x3 depthwise convolution
    2. Three parallel branches with strip convolutions:
       - Branch 1: 5x1 and 1x5 depthwise convolutions
       - Branch 2: 7x1 and 1x7 depthwise convolutions  
       - Branch 3: 9x1 and 1x9 depthwise convolutions
    3. Add all branch outputs with input
    4. Point-wise convolution (1x1 conv)
    5. Element-wise multiplication with input
    """
    
    def __init__(self, in_channels):
        super(SpatialCrosswiseConvModule, self).__init__()
        self.in_channels = in_channels
        
        # Initial 3x3 depthwise convolution
        self.square_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, padding=1, 
            groups=in_channels, bias=False
        )
        
        # Branch 1: 5x1 and 1x5 strip convolutions
        self.h_strip_conv_5 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(5, 1), padding=(2, 0),
            groups=in_channels, bias=False
        )
        self.v_strip_conv_5 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(1, 5), padding=(0, 2),
            groups=in_channels, bias=False
        )
        
        # Branch 2: 7x1 and 1x7 strip convolutions
        self.h_strip_conv_7 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(7, 1), padding=(3, 0),
            groups=in_channels, bias=False
        )
        self.v_strip_conv_7 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(1, 7), padding=(0, 3),
            groups=in_channels, bias=False
        )
        
        # Branch 3: 9x1 and 1x9 strip convolutions
        self.h_strip_conv_9 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(9, 1), padding=(4, 0),
            groups=in_channels, bias=False
        )
        self.v_strip_conv_9 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(1, 9), padding=(0, 4),
            groups=in_channels, bias=False
        )
        
        # Point-wise convolution (1x1 conv)
        self.pwconv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=1, bias=False
        )
        
        # Batch normalization layers
        self.bn_square = nn.BatchNorm2d(in_channels)
        self.bn_strip_5 = nn.BatchNorm2d(in_channels)
        self.bn_strip_7 = nn.BatchNorm2d(in_channels)
        self.bn_strip_9 = nn.BatchNorm2d(in_channels)
        self.bn_pwconv = nn.BatchNorm2d(in_channels)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature tensor with shape (B, C, H, W)
            
        Returns:
            Tensor: Output feature tensor with shape (B, C, H, W)
        """
        # Store input for residual connections
        identity = x
        
        # Initial 3x3 depthwise convolution
        x_square = self.relu(self.bn_square(self.square_conv(x)))
        
        # Branch 1: 5x1 and 1x5 strip convolutions
        branch_1 = self.h_strip_conv_5(x_square)
        branch_1 = self.v_strip_conv_5(branch_1)
        branch_1 = self.relu(self.bn_strip_5(branch_1))
        
        # Branch 2: 7x1 and 1x7 strip convolutions  
        branch_2 = self.h_strip_conv_7(x_square)
        branch_2 = self.v_strip_conv_7(branch_2)
        branch_2 = self.relu(self.bn_strip_7(branch_2))
        
        # Branch 3: 9x1 and 1x9 strip convolutions
        branch_3 = self.h_strip_conv_9(x_square)
        branch_3 = self.v_strip_conv_9(branch_3)
        branch_3 = self.relu(self.bn_strip_9(branch_3))
        
        # Add all branch outputs with module input
        branch_sum = branch_1 + branch_2 + branch_3 + identity
        
        # Point-wise convolution
        pwconv_out = self.relu(self.bn_pwconv(self.pwconv(branch_sum)))
        
        # Element-wise multiplication with module input
        output = pwconv_out * identity
        
        return output