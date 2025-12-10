#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用官方YOLOv12的YOWOv3训练脚本

这个脚本展示了如何使用官方YOLOv12替代自定义Level1实现进行训练。
主要优势:
1. 直接使用官方预训练权重
2. 完全兼容官方YOLOv12架构
3. 简化的训练流程
4. 更好的稳定性和性能

作者: Assistant
日期: 2024
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import time
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent / 'ultralytics'))

# 导入必要模块
try:
    from ultralytics import YOLO
    from integrate_official_yolov12 import create_yowov3_with_official_yolov12
    OFFICIAL_YOLO_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入必要模块: {e}")
    OFFICIAL_YOLO_AVAILABLE = False

try:
    from cus_datasets.ucf.build_dataset import UCF24Dataset
    from utils.loss import YOWOLoss
    from utils.EMA import ModelEMA
    from utils.warmup_lr import WarmupLRScheduler
except ImportError as e:
    print(f"警告: 无法导入YOWOv3模块: {e}")


class OfficialYOLOv12Trainer:
    """
    官方YOLOv12训练器
    
    使用官方YOLOv12架构进行YOWOv3模型训练
    """
    
    def __init__(self, config_path):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 初始化数据加载器
        self._init_dataloader()
        
        # 初始化优化器和调度器
        self._init_optimizer()
        
        # 初始化损失函数
        self._init_loss()
        
        # 初始化EMA
        if self.config['training'].get('use_ema', True):
            self.ema = ModelEMA(self.model)
        else:
            self.ema = None
        
        # 训练状态
        self.start_epoch = 0
        self.best_map = 0.0
        
        print("✓ 训练器初始化完成")
    
    def _init_model(self):
        """初始化模型"""
        print("初始化官方YOLOv12模型...")
        
        model_config = self.config['model']
        
        # 创建模型
        self.model = create_yowov3_with_official_yolov12(
            model_size=model_config['size'],
            num_classes=model_config['num_classes'],
            pretrained_path=model_config.get('pretrained_path')
        )
        
        # 加载预训练权重
        if model_config.get('pretrained_path'):
            pretrained_path = model_config['pretrained_path']
            if Path(pretrained_path).exists():
                print(f"加载预训练权重: {pretrained_path}")
                self.model.load_pretrained_2d(pretrained_path)
            else:
                print(f"预训练权重文件不存在: {pretrained_path}")
        
        # 冻结骨干网络 (可选)
        if model_config.get('freeze_backbone', False):
            print("冻结2D骨干网络")
            self.model.freeze_backbone_2d(True)
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 显示模型信息
        model_info = self.model.get_model_info()
        print(f"模型: {model_info['model_name']}")
        print(f"总参数: {model_info['total_parameters']:,}")
        print(f"可训练参数: {model_info['trainable_parameters']:,}")
    
    def _init_dataloader(self):
        """初始化数据加载器"""
        print("初始化数据加载器...")
        
        data_config = self.config['data']
        
        # 训练数据集
        train_dataset = UCF24Dataset(
            data_root=data_config['train_path'],
            img_size=data_config['img_size'],
            clip_len=data_config['clip_len'],
            is_train=True
        )
        
        # 验证数据集
        val_dataset = UCF24Dataset(
            data_root=data_config['val_path'],
            img_size=data_config['img_size'],
            clip_len=data_config['clip_len'],
            is_train=False
        )
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=True
        )
        
        print(f"训练样本: {len(train_dataset)}")
        print(f"验证样本: {len(val_dataset)}")
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        train_config = self.config['training']
        
        # 优化器
        if train_config['optimizer']['type'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=train_config['optimizer']['lr'],
                weight_decay=train_config['optimizer']['weight_decay']
            )
        elif train_config['optimizer']['type'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=train_config['optimizer']['lr'],
                momentum=train_config['optimizer'].get('momentum', 0.9),
                weight_decay=train_config['optimizer']['weight_decay']
            )
        
        # 学习率调度器
        if train_config['lr_scheduler']['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs']
            )
        elif train_config['lr_scheduler']['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config['lr_scheduler']['step_size'],
                gamma=train_config['lr_scheduler']['gamma']
            )
        
        # Warmup调度器
        if train_config['lr_scheduler'].get('warmup_epochs', 0) > 0:
            self.warmup_scheduler = WarmupLRScheduler(
                self.optimizer,
                warmup_epochs=train_config['lr_scheduler']['warmup_epochs']
            )
        else:
            self.warmup_scheduler = None
    
    def _init_loss(self):
        """初始化损失函数"""
        loss_config = self.config['training']['loss']
        
        self.criterion = YOWOLoss(
            num_classes=self.config['model']['num_classes'],
            cls_loss_weight=loss_config['cls_loss_weight'],
            box_loss_weight=loss_config['box_loss_weight'],
            obj_loss_weight=loss_config.get('obj_loss_weight', 1.0)
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images_2d, videos_3d, targets) in enumerate(pbar):
            # 数据移动到设备
            images_2d = images_2d.to(self.device)
            videos_3d = videos_3d.to(self.device)
            targets = [t.to(self.device) for t in targets]
            
            # 前向传播
            outputs = self.model(images_2d, videos_3d)
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # 更新EMA
            if self.ema:
                self.ema.update(self.model)
            
            # 更新Warmup调度器
            if self.warmup_scheduler and epoch < self.config['training']['lr_scheduler']['warmup_epochs']:
                self.warmup_scheduler.step()
            
            # 统计
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 更新学习率调度器
        if epoch >= self.config['training']['lr_scheduler'].get('warmup_epochs', 0):
            self.scheduler.step()
        
        return avg_loss
    
    def validate(self, epoch):
        """验证模型"""
        model_to_eval = self.ema.ema if self.ema else self.model
        model_to_eval.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, (images_2d, videos_3d, targets) in enumerate(pbar):
                # 数据移动到设备
                images_2d = images_2d.to(self.device)
                videos_3d = videos_3d.to(self.device)
                targets = [t.to(self.device) for t in targets]
                
                # 前向传播
                outputs = model_to_eval(images_2d, videos_3d)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'Val Loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        avg_val_loss = total_loss / num_batches
        return avg_val_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'config': self.config
        }
        
        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.ema.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'last.pt')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
            print(f"✓ 保存最佳模型 (epoch {epoch+1})")
    
    def train(self):
        """开始训练"""
        print("\n=== 开始训练 ===")
        print(f"模型: {self.config['model']['type']}")
        print(f"数据集: {self.config['data']['dataset']}")
        print(f"训练轮数: {self.config['training']['epochs']}")
        print(f"批次大小: {self.config['data']['batch_size']}")
        print(f"学习率: {self.config['training']['optimizer']['lr']}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate(epoch)
            
            # 打印结果
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            is_best = val_loss < self.best_map  # 这里应该用mAP，简化为用loss
            if is_best:
                self.best_map = val_loss
            
            if (epoch + 1) % self.config['training']['save_period'] == 0:
                self.save_checkpoint(epoch, is_best)
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n=== 训练完成 ===")
        print(f"总耗时: {total_time/3600:.2f} 小时")
        print(f"最佳验证损失: {self.best_map:.4f}")
        print(f"模型保存路径: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='官方YOLOv12训练脚本')
    parser.add_argument('--config', type=str, default='config/official_yolov12.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not OFFICIAL_YOLO_AVAILABLE:
        print("❌ 错误: 无法导入官方YOLOv12模块")
        print("请安装ultralytics: pip install ultralytics")
        return
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        return
    
    try:
        # 创建训练器
        trainer = OfficialYOLOv12Trainer(args.config)
        
        # 恢复训练 (如果指定)
        if args.resume and Path(args.resume).exists():
            print(f"恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.start_epoch = checkpoint['epoch'] + 1
            trainer.best_map = checkpoint['best_map']
            
            if trainer.ema and 'ema_state_dict' in checkpoint:
                trainer.ema.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()