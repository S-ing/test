# -*- coding: utf-8 -*-
"""
YOLOv12 Level1 增强版本实现

这是YOLOv12的Level1增强版本，在Level0基础上添加了：
- C3k2模块：增强的CSP模块，具有更好的特征提取能力
- SEAttention：轻量级通道注意力机制
- 保持Level0的基础架构，逐步增强模型能力

相比Level0的改进：
- 使用C3k2替代部分C3k模块，提升特征提取能力
- 在关键位置添加SEAttention，增强通道特征表达
- 保持模型轻量化，参数增长控制在合理范围内

目的：
- 验证增强模块对性能的提升效果
- 为后续更复杂组件的添加提供基础
- 保持训练稳定性的同时提升模型性能

Author: YOLOv12 Level1 Implementation
Date: 2025.6.10
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目根目录到路径以支持绝对导入
try:
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取YOWOv3项目根目录（scripts的上级目录）
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Added project root to path: {project_root}")
except Exception as e:
    print(f"Error setting up path: {e}")

# 导入DFL检测头模块
try:
    from model.head.dfl import DFLHead
    print("Successfully imported DFLHead")
except ImportError as e:
    print(f"Warning: Could not import DFLHead: {e}")
    print("This may affect some model components, but training can continue")


def pad(k, p=None, d=1):
    """计算卷积层的填充大小"""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    """融合卷积层和批归一化层以提升推理速度"""
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           kernel_size=conv.kernel_size,
                           stride=conv.stride,
                           padding=conv.padding,
                           groups=conv.groups,
                           bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(nn.Module):
    """标准卷积模块"""

    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.act(self.conv(x))


class Residual(nn.Module):
    """残差模块"""

    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = nn.Sequential(Conv(ch, ch, 3), Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class SEAttention(nn.Module):
    """SE注意力机制 - Level1新增组件

    轻量级通道注意力机制，通过全局平均池化和两个全连接层
    学习通道间的重要性权重，提升特征表达能力。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class C3k(nn.Module):
    """基础C3k模块 - 从Level0继承"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Residual(c_, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    """增强的C3k2模块 - Level1新增组件

    相比C3k的改进：
    - 增加了更多的残差连接路径
    - 更好的特征融合机制
    - 可选的注意力机制集成
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, use_se=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # 增强的残差块序列
        self.m = nn.Sequential(*(Residual(c_, shortcut) for _ in range(n)))

        # 可选的SE注意力机制
        self.use_se = use_se
        if use_se:
            self.se = SEAttention(c2)

        # 额外的特征融合路径
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        # 主要的特征提取路径
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)

        # 特征融合
        out = self.cv3(torch.cat((y1, y2), 1))

        # 残差连接
        if self.shortcut:
            out = out + x

        # SE注意力
        if self.use_se:
            out = self.se(out)

        return out


class C3k2_SE(nn.Module):
    """集成SE注意力的C3k2模块 - Level1组合组件"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        self.c3k2 = C3k2(c1, c2, n, shortcut, g, e, k, use_se=True)

    def forward(self, x):
        return self.c3k2(x)


class SPPF(nn.Module):
    """空间金字塔池化模块(SPPF) - 从Level0继承"""

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.cv1 = Conv(in_ch, in_ch // 2)
        self.cv2 = Conv(in_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat([x, y1, y2, self.pool(y2)], 1))


class YOLOv12Backbone_Level1(nn.Module):
    """YOLOv12主干网络 - Level1增强版本

    Level1版本特点：
    - 在关键位置使用C3k2模块替代C3k
    - 添加SEAttention增强通道特征表达
    - 保持Level0的基础架构，逐步增强
    - 参数增长控制在合理范围内
    """

    def __init__(self, depth_multiple=0.33, width_multiple=0.25, pretrained=None):
        super().__init__()
        
        # 保存预训练路径供load_pretrain方法使用
        self.pretrained = pretrained

        # Level1: 使用与Level0相同的通道配置，但增强模块能力
        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor

        # 通道配置与Level0保持一致
        base_channels = [64, 128, 256, 512, 1024]
        self.channels = [make_divisible(ch * width_multiple) for ch in base_channels]

        # 计算深度参数
        def get_depth(n):
            return max(round(n * depth_multiple), 1)

        # Stage 0: 输入处理
        self.stem = Conv(3, self.channels[0], 6, 2, 2)  # 640 -> 320

        # Stage 1: 第一个下采样阶段 - 使用基础C3k
        self.stage1 = nn.Sequential(
            Conv(self.channels[0], self.channels[1], 3, 2),  # 320 -> 160
            C3k(self.channels[1], self.channels[1], get_depth(3), True)
        )

        # Stage 2: 第二个下采样阶段 - 开始使用C3k2增强
        self.stage2 = nn.Sequential(
            Conv(self.channels[1], self.channels[2], 3, 2),  # 160 -> 80
            C3k2(self.channels[2], self.channels[2], get_depth(6), True, use_se=True)  # Level1增强
        )

        # Stage 3: 第三个下采样阶段 - 使用C3k2_SE
        self.stage3 = nn.Sequential(
            Conv(self.channels[2], self.channels[3], 3, 2),  # 80 -> 40
            C3k2_SE(self.channels[3], self.channels[3], get_depth(6), True)  # Level1增强
        )

        # Stage 4: 第四个下采样阶段 - 最强增强 + SPPF
        self.stage4_conv = Conv(self.channels[3], self.channels[4], 3, 2)  # 40 -> 20
        self.stage4_c3k2 = C3k2_SE(self.channels[4], self.channels[4], get_depth(3), True)  # Level1增强
        self.stage4_sppf = SPPF(self.channels[4], self.channels[4])  # 保留SPPF

        # 额外的SE注意力用于最终特征增强
        self.final_se = SEAttention(self.channels[4])

    def forward(self, x):
        """前向传播，返回多尺度特征"""
        x = self.stem(x)  # Stage 0
        x = self.stage1(x)  # Stage 1

        p3 = self.stage2(x)  # Stage 2 - P3特征 (80x80) - Level1增强
        p4 = self.stage3(p3)  # Stage 3 - P4特征 (40x40) - Level1增强

        # Stage 4 - P5特征 (20x20) - Level1最强增强
        p5 = self.stage4_conv(p4)
        p5 = self.stage4_c3k2(p5)
        p5 = self.stage4_sppf(p5)
        p5 = self.final_se(p5)  # 最终SE增强

        return p3, p4, p5

    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重 - Level1版本"""
        try:
            print(f"Loading Level1 backbone pretrained weights from: {pretrained_path}")

            if pretrained_path.startswith('http'):
                checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location='cpu')
            else:
                checkpoint = torch.load(pretrained_path, map_location='cpu')

            if 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint

            # 处理DetectionModel对象
            if hasattr(pretrained_dict, 'state_dict'):
                pretrained_dict = pretrained_dict.state_dict()
            elif not isinstance(pretrained_dict, dict):
                print(f"⚠️  Unexpected pretrained dict type: {type(pretrained_dict)}")
                print("Attempting to extract state_dict...")
                if hasattr(pretrained_dict, '__dict__'):
                    pretrained_dict = pretrained_dict.__dict__
                else:
                    print("Cannot extract state_dict, skipping pretrained weights")
                    return

            # 获取当前模型的state_dict
            model_dict = self.state_dict()

            # Level1特殊处理：尝试从Level0权重中加载兼容的部分
            filtered_dict = {}
            matched_keys = 0
            size_mismatches = []

            for k, v in pretrained_dict.items():
                # 直接匹配
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                        matched_keys += 1
                    else:
                        size_mismatches.append((k, model_dict[k].shape, v.shape))
                # 尝试从C3k映射到C3k2（基础部分）
                elif 'c3k2' in k:
                    # 尝试找到对应的C3k权重
                    c3k_key = k.replace('c3k2', 'c3k')
                    if c3k_key in pretrained_dict:
                        if pretrained_dict[c3k_key].shape == model_dict[k].shape:
                            filtered_dict[k] = pretrained_dict[c3k_key]
                            matched_keys += 1

            print(f"Level1 Backbone - Successfully matched: {matched_keys} layers")
            if size_mismatches:
                print(f"Level1 Backbone - Size mismatches: {len(size_mismatches)}")

            # 更新模型权重
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict, strict=False)

            if matched_keys > 0:
                print(f"✅ Level1 Backbone - Successfully loaded {matched_keys} layers from pretrained weights")
            else:
                print("⚠️  Level1 Backbone - No matching layers found, training from scratch")

        except Exception as e:
            print(f"❌ Level1 Backbone - Error loading pretrained weights: {str(e)}")
            print("Continuing without pretrained weights...")

    def load_pretrain(self):
        """YOWOv3框架要求的预训练加载接口"""
        if hasattr(self, 'pretrained') and self.pretrained and self.pretrained != 'None':
            self.load_pretrained_weights(self.pretrained)
        else:
            print("No pretrained weights specified for YOLOv12 Level1")


class YOLOv12NeckHead_Level1(nn.Module):
    """YOLOv12颈部和检测头 - Level1增强版本

    Level1版本特点：
    - 在FPN路径中使用C3k2模块
    - 添加SE注意力增强特征融合
    - 保持基础FPN结构的同时提升性能
    """

    def __init__(self, backbone_channels, num_classes=80, depth_multiple=0.33):
        super().__init__()

        def get_depth(n):
            return max(round(n * depth_multiple), 1)

        # 从backbone获取通道数
        p3_ch, p4_ch, p5_ch = backbone_channels

        # Top-down pathway (自顶向下路径) - Level1增强
        self.upsample = nn.Upsample(None, 2, 'nearest')

        # P5 -> P4 融合 - 使用C3k2增强
        self.reduce_p5 = Conv(p5_ch, p4_ch, 1, 1)
        self.c3k2_p4 = C3k2_SE(p4_ch + p4_ch, p4_ch, get_depth(3), False)  # Level1增强

        # P4 -> P3 融合 - 使用C3k2增强
        self.reduce_p4 = Conv(p4_ch, p3_ch, 1, 1)
        self.c3k2_p3 = C3k2_SE(p3_ch + p3_ch, p3_ch, get_depth(3), False)  # Level1增强

        # Bottom-up pathway (自底向上路径) - Level1增强
        # P3 -> P4 融合
        self.downsample_p3 = Conv(p3_ch, p3_ch, 3, 2)
        self.c3k2_p4_out = C3k2_SE(p3_ch + p4_ch, p4_ch, get_depth(3), False)  # Level1增强

        # P4 -> P5 融合
        self.downsample_p4 = Conv(p4_ch, p4_ch, 3, 2)
        self.c3k2_p5_out = C3k2_SE(p4_ch + p5_ch, p5_ch, get_depth(3), False)  # Level1增强

        # 输出特征的SE注意力增强
        self.se_p3 = SEAttention(p3_ch)
        self.se_p4 = SEAttention(p4_ch)
        self.se_p5 = SEAttention(p5_ch)

        # 检测头
        try:
            self.head = DFLHead(num_classes, [p3_ch, p4_ch, p5_ch])
            self.use_dfl_head = True
        except:
            print("Warning: DFLHead not available, using simple detection head")
            self.head = nn.ModuleList([
                nn.Conv2d(p3_ch, num_classes + 4, 1),
                nn.Conv2d(p4_ch, num_classes + 4, 1),
                nn.Conv2d(p5_ch, num_classes + 4, 1)
            ])
            self.use_dfl_head = False

    def forward(self, features):
        """前向传播：增强的FPN特征融合 + 检测头"""
        p3, p4, p5 = features

        # Top-down pathway - Level1增强
        # P5 -> P4
        p5_up = self.upsample(self.reduce_p5(p5))
        p4_fused = self.c3k2_p4(torch.cat([p4, p5_up], 1))  # 使用C3k2_SE

        # P4 -> P3
        p4_up = self.upsample(self.reduce_p4(p4_fused))
        p3_out = self.c3k2_p3(torch.cat([p3, p4_up], 1))  # 使用C3k2_SE

        # Bottom-up pathway - Level1增强
        # P3 -> P4
        p3_down = self.downsample_p3(p3_out)
        p4_out = self.c3k2_p4_out(torch.cat([p4_fused, p3_down], 1))  # 使用C3k2_SE

        # P4 -> P5
        p4_down = self.downsample_p4(p4_out)
        p5_out = self.c3k2_p5_out(torch.cat([p5, p4_down], 1))  # 使用C3k2_SE

        # SE注意力增强输出特征
        p3_out = self.se_p3(p3_out)
        p4_out = self.se_p4(p4_out)
        p5_out = self.se_p5(p5_out)

        # 检测头
        if self.use_dfl_head:
            return self.head([p3_out, p4_out, p5_out])
        else:
            # 简单的检测头实现
            return [head(feat) for head, feat in zip(self.head, [p3_out, p4_out, p5_out])]


class YOLO_Level1(nn.Module):
    """YOLOv12完整模型 - Level1增强版本

    Level1版本特点：
    - 在Level0基础上添加C3k2和SEAttention
    - 保持基础架构稳定性
    - 逐步增强模型性能
    - 参数增长控制在合理范围内
    """

    def __init__(self, num_classes=80, depth_multiple=0.33, width_multiple=0.25, pretrained=None):
        super().__init__()

        print(f"Initializing YOLOv12 Level1 with depth_multiple={depth_multiple}, width_multiple={width_multiple}")

        # 初始化backbone
        self.backbone = YOLOv12Backbone_Level1(depth_multiple, width_multiple)

        # 获取backbone输出通道数
        backbone_channels = self.backbone.channels[-3:]  # P3, P4, P5的通道数

        # 初始化neck和head
        self.neck_head = YOLOv12NeckHead_Level1(backbone_channels, num_classes, depth_multiple)

        # 为YOWOv3框架兼容性添加detection_head属性
        # 指向neck_head中的检测头部分
        if hasattr(self.neck_head, 'head') and hasattr(self.neck_head.head, 'stride'):
            self.detection_head = self.neck_head.head
        else:
            # 如果没有DFLHead，创建一个简单的兼容对象
            class SimpleDetectionHead:
                def __init__(self, nc, no):
                    self.nc = nc  # number of classes
                    self.no = no  # number of outputs per anchor
                    self.stride = torch.tensor([8., 16., 32.])  # 默认stride

                    # 创建一个简单的DFL对象用于兼容性
                    class SimpleDFL:
                        def __init__(self):
                            self.ch = 16  # DFL channels

                    self.dfl = SimpleDFL()

            self.detection_head = SimpleDetectionHead(num_classes, num_classes + 4 * 16)

        # 保存配置
        self.num_classes = num_classes
        self.pretrained = pretrained

        # 设置输出通道数（YOWOv3框架要求）
        self.out_channels = backbone_channels  # [P3_ch, P4_ch, P5_ch]

        # 加载预训练权重
        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, x):
        """前向传播"""
        # Backbone特征提取
        features = self.backbone(x)

        # 对于YOWOv3集成，只返回backbone特征，不进行检测
        # YOWOv3框架会处理后续的检测头
        return features

    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重 - Level1版本"""
        try:
            print(f"Loading Level1 pretrained weights from: {pretrained_path}")

            if pretrained_path.startswith('http'):
                checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location='cpu')
            else:
                checkpoint = torch.load(pretrained_path, map_location='cpu')

            if 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint

            # 处理DetectionModel对象
            if hasattr(pretrained_dict, 'state_dict'):
                pretrained_dict = pretrained_dict.state_dict()
            elif not isinstance(pretrained_dict, dict):
                print(f"⚠️  Unexpected pretrained dict type: {type(pretrained_dict)}")
                print("Attempting to extract state_dict...")
                if hasattr(pretrained_dict, '__dict__'):
                    pretrained_dict = pretrained_dict.__dict__
                else:
                    print("Cannot extract state_dict, skipping pretrained weights")
                    return

            # 获取当前模型的state_dict
            model_dict = self.state_dict()

            # Level1特殊处理：尝试从Level0权重中加载兼容的部分
            filtered_dict = {}
            matched_keys = 0
            size_mismatches = []

            for k, v in pretrained_dict.items():
                # 直接匹配
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                        matched_keys += 1
                    else:
                        size_mismatches.append((k, model_dict[k].shape, v.shape))
                # 尝试从C3k映射到C3k2（基础部分）
                elif 'c3k2' in k:
                    # 尝试找到对应的C3k权重
                    c3k_key = k.replace('c3k2', 'c3k')
                    if c3k_key in pretrained_dict:
                        if pretrained_dict[c3k_key].shape == model_dict[k].shape:
                            filtered_dict[k] = pretrained_dict[c3k_key]
                            matched_keys += 1

            print(f"Level1 - Successfully matched: {matched_keys} layers")
            if size_mismatches:
                print(f"Level1 - Size mismatches: {len(size_mismatches)}")

            # 更新模型权重
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict, strict=False)

            if matched_keys > 0:
                print(f"✅ Level1 - Successfully loaded {matched_keys} layers from pretrained weights")
            else:
                print("⚠️  Level1 - No matching layers found, training from scratch")

        except Exception as e:
            print(f"❌ Level1 - Error loading pretrained weights: {str(e)}")
            print("Continuing without pretrained weights...")

    def load_pretrain(self):
        """YOWOv3框架要求的预训练加载接口"""
        if hasattr(self, 'pretrained') and self.pretrained and self.pretrained != 'None':
            self.load_pretrained_weights(self.pretrained)
        else:
            print("No pretrained weights specified for YOLOv12 Level1")

    def fuse(self):
        """融合Conv2d + BatchNorm2d层以优化推理"""
        print("Fusing YOLOv12 Level1 layers...")
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                delattr(m, 'norm')
                m.forward = m.fuse_forward
        return self


def build_yolov12_level1(version='n', num_classes=80, pretrained=None):
    """构建YOLOv12 Level1模型

    Level1版本配置：
    - 在Level0基础上添加C3k2和SEAttention
    - 保持基础架构稳定性
    - 逐步增强模型性能
    - 适合验证增强组件的效果

    Args:
        version (str): 模型版本 ('n', 's', 'm', 'l', 'x')
        num_classes (int): 检测类别数量
        pretrained (str): 预训练权重路径

    Returns:
        YOLO_Level1: Level1版本的YOLOv12模型
    """
    # Level1版本使用与Level0相同的基础配置，但增强模块能力
    version_configs = {
        'n': {'depth_multiple': 0.33, 'width_multiple': 0.25},  # 增强版nano
        's': {'depth_multiple': 0.33, 'width_multiple': 0.50},  # 增强版small
        'm': {'depth_multiple': 0.67, 'width_multiple': 0.75},  # 增强版medium
        'l': {'depth_multiple': 1.00, 'width_multiple': 1.00},  # 增强版large
        'x': {'depth_multiple': 1.33, 'width_multiple': 1.25},  # 增强版extra large
    }

    if version not in version_configs:
        raise ValueError(f"Unsupported version: {version}. Choose from {list(version_configs.keys())}")

    config = version_configs[version]
    depth_multiple = config['depth_multiple']
    width_multiple = config['width_multiple']

    print(f"Building YOLOv12 Level1-{version} with depth_multiple={depth_multiple}, width_multiple={width_multiple}")

    model = YOLO_Level1(
        num_classes=num_classes,
        depth_multiple=depth_multiple,
        width_multiple=width_multiple,
        pretrained=pretrained
    )

    return model


# 导入训练相关模块
import argparse
import yaml
import time
from torch.utils import data
from cus_datasets.build_dataset import build_dataset
from cus_datasets.collate_fn import collate_fn
from model.TSN.YOWOv3 import build_yowov3 
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
from utils.EMA import EMA
import shutil
from utils.flops import get_info


def train_level1_model(config):
    """
    Train YOLOv12 Level1 model with YOWOv3 framework
    
    Args:
        config (dict): Configuration dictionary containing all training parameters
    """
    
    print("🚀 Starting YOLOv12 Level1 Training...")
    print(f"📁 Save folder: {config['save_folder']}")
    print(f"🎯 Dataset: {config['dataset']}")
    print(f"📊 Batch size: {config['batch_size']}")
    print(f"🔄 Max epochs: {config['max_epoch']}")
    
    # Create save directory
    os.makedirs(config['save_folder'], exist_ok=True)
    
    # Save config file
    #######################################################
    source_file = config['config_path']
    destination_file = os.path.join(config['save_folder'], 'config.yaml')
    shutil.copyfile(source_file, destination_file)
    print(f"💾 Config saved to: {destination_file}")
    #######################################################
    
    # Create dataloader, model, criterion
    ####################################################
    print("📚 Building dataset...")
    dataset = build_dataset(config, phase='train')
    
    dataloader = data.DataLoader(dataset, config['batch_size'], True, collate_fn=collate_fn,
                                 num_workers=config['num_workers'], pin_memory=True)
    
    print("🏗️  Building YOLOv12 Level1 model...")
    model = build_yowov3(config)
    
    # Get model information
    get_info(config, model)
    
    # Print Level1 specific information
    if hasattr(model.net2D, 'backbone'):
        backbone_params = sum(p.numel() for p in model.net2D.backbone.parameters())
        print(f"🧠 Level1 Backbone parameters: {backbone_params:,}")
    
    model.to("cuda")
    model.train()
    
    print("🎯 Building loss function...")
    criterion = build_loss(model, config)
    #####################################################

    # Optimizer setup with parameter grouping
    print("⚙️  Setting up optimizer...")
    
    # Parameter grouping for different learning rates
    backbone_params = []
    neck_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name or 'net2D' in name:
            backbone_params.append(param)
        elif 'neck' in name or 'fpn' in name:
            neck_params.append(param)
        else:
            head_params.append(param)
    
    # Different learning rates for different parts
    param_groups = [
        {'params': backbone_params, 'lr': config['lr'] * 0.1},  # Lower LR for backbone
        {'params': neck_params, 'lr': config['lr']},
        {'params': head_params, 'lr': config['lr']}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, 
                                  lr=config['lr'], 
                                  weight_decay=config['weight_decay'])
    
    # Training parameters
    adjustlr_schedule = config['adjustlr_schedule']
    acc_grad = config['acc_grad']
    lr_decay = config['lr_decay']
    
    # Warmup scheduler
    warmup_lr = LinearWarmup(config)
    
    print(f"📈 Optimizer: AdamW with {len(param_groups)} parameter groups")
    print(f"🔥 Learning rate: {config['lr']} (backbone: {config['lr'] * 0.1})")
    print(f"🌡️  Warmup steps: {config['max_step_warmup']}")
    
    # Initialize training variables
    cnt_pram_update = 0
    ema = EMA(model)
    
    # Training loop
    print("\n🎯 Starting Level1 training loop...")
    
    best_map = 0.0
    start_time = time.time()
    
    for epoch in range(config['max_epoch']):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        loss_acc = 0.0
        num_batches = 0
        
        print(f"\n📅 Epoch [{epoch+1}/{config['max_epoch']}]")
        
        for batch_idx, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader):
            # Move to GPU
            batch_size = batch_clip.shape[0]
            batch_clip = batch_clip.cuda()
            
            for idx in range(batch_size):
                batch_bboxes[idx] = batch_bboxes[idx].cuda()
                batch_labels[idx] = batch_labels[idx].cuda()
            
            # Forward pass
            outputs = model(batch_clip)
            
            # Build targets
            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                nbox = bboxes.shape[0]
                nclass = labels.shape[1]
                target = torch.Tensor(nbox, 5 + nclass)
                target[:, 0] = i
                target[:, 1:5] = bboxes
                target[:, 5:] = labels
                targets.append(target)
            
            targets = torch.cat(targets, dim=0)
            
            # Compute loss
            loss = criterion(outputs, targets) / acc_grad
            total_loss = loss
            
            # Backward pass
            loss_acc += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            loss.backward()
            
            if (batch_idx + 1) % acc_grad == 0:
                cnt_pram_update = cnt_pram_update + 1
                if epoch == 0:  # First epoch warmup
                    warmup_lr(optimizer, cnt_pram_update)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)
            
            # Print progress
            if (batch_idx + 1) % acc_grad == 0:
                current_lr = optimizer.param_groups[0]['lr']
                loss_acc += loss.item()
                print(f"Epoch: {epoch + 1}, Update: {cnt_pram_update}, Loss: {loss_acc:.6f}, LR: {current_lr:.8f}", flush=True)
                
                # Log to file
                with open(os.path.join(config['save_folder'], "training_log.txt"), "a") as f:
                    f.write(f"Epoch: {epoch + 1}, Update: {cnt_pram_update}, Loss: {loss_acc:.6f}, LR: {current_lr:.8f}\n")

                loss_acc = 0.0
        
        # Learning rate scheduling (after warmup)
        if epoch >= 1 and (epoch + 1) in adjustlr_schedule:
            old_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            new_lr = optimizer.param_groups[0]['lr']
            print(f"     📉 Learning rate adjusted: {old_lr:.8f} -> {new_lr:.8f}")
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        print(f"  📊 Epoch {epoch+1} Summary:")
        print(f"     Average Loss: {avg_loss:.4f}")
        print(f"     Time: {epoch_time:.2f}s")
        print(f"     LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint every epoch (like Level0)
        # Save EMA model
        ema_path = os.path.join(config['save_folder'], f'level1_ema_epoch_{epoch+1}.pth')
        torch.save(ema.ema.state_dict(), ema_path)
        
        # Save regular model
        checkpoint_path = os.path.join(config['save_folder'], f'level1_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, checkpoint_path)
        print(f"     💾 Model saved: {checkpoint_path}")
        print(f"     💾 EMA model saved: {ema_path}")
    
    # Save final model
    final_model_path = os.path.join(config['save_folder'], 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_completed': True
    }, final_model_path)
    
    total_time = time.time() - start_time
    print(f"\n✅ Level1 training completed!")
    print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
    print(f"💾 Final model saved: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv12 Level1 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--test_only', action='store_true', help='Only test model without training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add config path to config dict
    config['config_path'] = args.config
    
    if args.test_only:
        # 测试Level1模型
        print("Testing YOLOv12 Level1 model...")

        # 创建模型
        model = build_yolov12_level1('n', num_classes=80)

        # 测试前向传播
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)
            print(f"Input shape: {x.shape}")
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                for i, output in enumerate(outputs):
                    print(f"Output {i} shape: {output.shape}")
            else:
                print(f"Output shape: {outputs.shape}")

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

        # 与Level0对比
        try:
            from YOLOv12_Level0 import build_yolov12_level0

            level0_model = build_yolov12_level0('n', num_classes=80)
            level0_params = sum(p.numel() for p in level0_model.parameters())

            param_increase = (total_params - level0_params) / level0_params * 100
            print(f"\nComparison with Level0:")
            print(f"Level0 parameters: {level0_params:,}")
            print(f"Level1 parameters: {total_params:,}")
            print(f"Parameter increase: {param_increase:.2f}%")

        except ImportError:
            print("\nCannot import Level0 for comparison")

        print("\n✅ YOLOv12 Level1 model test completed successfully!")
    else:
        # Start training
        train_level1_model(config)