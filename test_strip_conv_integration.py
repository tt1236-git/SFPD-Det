#!/usr/bin/env python3
"""
测试条形卷积模块集成的脚本

这个脚本用于验证条形卷积模块是否正确集成到网络中。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.abspath('.'))

def test_spatial_crosswise_conv_module():
    """测试条形卷积模块"""
    print("测试条形卷积模块...")
    
    try:
        from mmrotate.models.roi_heads.bbox_heads.spatial_crosswise_conv_module import SpatialCrosswiseConvModule
        
        # 创建模块实例
        in_channels = 512
        module = SpatialCrosswiseConvModule(in_channels)
        
        # 创建测试输入
        batch_size = 2
        height, width = 7, 7
        x = torch.randn(batch_size, in_channels, height, width)
        
        # 前向传播
        output = module(x)
        
        # 检查输出形状
        assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
        
        print(f"✓ 条形卷积模块测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 条形卷积模块测试失败: {e}")
        return False

def test_layer_reg_with_strip_conv():
    """测试带条形卷积的LayerReg"""
    print("\n测试带条形卷积的LayerReg...")
    
    try:
        from mmrotate.models.roi_heads.bbox_heads.layer_reg_with_strip_conv import LayerRegWithStripConv
        
        # 创建模块实例
        in_channels = 512
        out_channels = 5
        num_convs = 3
        feat_size = 7
        
        layer_reg = LayerRegWithStripConv(
            in_channels=in_channels,
            out_channels=out_channels, 
            num_convs=num_convs,
            feat_size=feat_size
        )
        
        # 创建测试输入 (B, N, C)
        batch_size = 2
        num_patches = feat_size * feat_size  # 49
        x = torch.randn(batch_size, num_patches, in_channels)
        
        # 前向传播
        output = layer_reg(x)
        
        # 检查输出形状
        expected_shape = (batch_size, out_channels)
        assert output.shape == expected_shape, f"输出形状不匹配: {output.shape} vs {expected_shape}"
        
        print(f"✓ 带条形卷积的LayerReg测试通过")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 带条形卷积的LayerReg测试失败: {e}")
        return False

def test_block_stdc_integration():
    """测试BlockSTDC集成"""
    print("\n测试BlockSTDC集成...")
    
    try:
        from mmrotate.models.roi_heads.bbox_heads.activation_mask_stdc import BlockSTDC
        
        # 创建BlockSTDC实例
        dim = 512
        num_heads = 8
        num_convs = 3
        
        block = BlockSTDC(
            dim=dim,
            num_heads=num_heads,
            num_convs=num_convs,
            dc_mode_str='XY',
            am_mode_str='V'
        )
        
        print(f"✓ BlockSTDC集成成功")
        print(f"  使用的LayerReg类型: {type(block.layer_reg).__name__}")
        
        # 检查是否使用了新的LayerReg
        if hasattr(block, 'layer_reg'):
            if 'WithStripConv' in type(block.layer_reg).__name__:
                print(f"✓ 确认使用了带条形卷积的LayerReg")
                return True
            else:
                print(f"✗ 仍使用旧的LayerReg")
                return False
        else:
            print(f"✗ 没有layer_reg属性")
            return False
        
    except Exception as e:
        print(f"✗ BlockSTDC集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("条形卷积模块集成测试")
    print("=" * 60)
    
    tests = [
        test_spatial_crosswise_conv_module,
        test_layer_reg_with_strip_conv,
        test_block_stdc_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    test_names = [
        "条形卷积模块",
        "带条形卷积的LayerReg", 
        "BlockSTDC集成"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(results)
    print(f"\n总体结果: {'✓ 所有测试通过' if all_passed else '✗ 存在测试失败'}")
    
    if all_passed:
        print("\n🎉 恭喜！条形卷积模块已成功集成到您的网络中！")
        print("\n现在您可以使用原来的训练和测试命令：")
        print("训练命令:")
        print("python ./tools/train.py ./configs/rotated_sfpd_det/dior-r/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_diorr_ms_rr_le90_xyawh321v.py")
        print("\n测试命令:")
        print("python ./tools/test.py ./configs/rotated_sfpd_det/dior-r/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_diorr_ms_rr_le90_xyawh321v.py ./work_dirs/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_diorr_ms_rr_le90_stdc_xyawh321v/xxx.pth --eval mAP")
    else:
        print("\n❌ 集成存在问题，请检查错误信息并修复。")
    
    return all_passed

if __name__ == "__main__":
    main()