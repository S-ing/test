#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12 Level0 Training Script for YOWOv3

This script is specifically designed for training the YOLOv12 Level0 model
within the YOWOv3 framework. It includes optimizations and configurations
tailored for the simplified Level0 architecture.

Usage:
    python scripts/train_level0.py --config config/yolov12_level0.yaml
"""

import os
import sys

# Get the absolute path to the YOWOv3 directory
script_dir = os.path.dirname(os.path.abspath(__file__))
yowov3_dir = os.path.dirname(script_dir)

# Add YOWOv3 directory to Python path
if yowov3_dir not in sys.path:
    sys.path.insert(0, yowov3_dir)

# Change working directory to YOWOv3
os.chdir(yowov3_dir)

import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob
import argparse

from math import sqrt
try:
    from utils.gradflow_check import plot_grad_flow
except ImportError:
    # Fallback: define a dummy function if import fails
    def plot_grad_flow(*args, **kwargs):
        pass
        
from utils.EMA import EMA
import logging
from utils.build_config import build_config
from cus_datasets.ucf.load_data import UCF_dataset
from cus_datasets.collate_fn import collate_fn
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3 
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
import shutil
from utils.flops import get_info


def train_level0_model(config):
    """
    Train YOLOv12 Level0 model with YOWOv3 framework
    
    Args:
        config (dict): Configuration dictionary containing all training parameters
    """
    
    print("🚀 Starting YOLOv12 Level0 Training...")
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
    
    print("🏗️  Building YOLOv12 Level0 model...")
    model = build_yowov3(config)
    
    # Get model information
    get_info(config, model)
    
    # Print Level0 specific information
    if hasattr(model.net2D, 'backbone'):
        backbone_params = sum(p.numel() for p in model.net2D.backbone.parameters())
        print(f"🧠 Level0 Backbone parameters: {backbone_params:,}")
    
    model.to("cuda")
    model.train()
    
    print("🎯 Building loss function...")
    criterion = build_loss(model, config)
    #####################################################

    # Optimizer setup with parameter grouping
    print("⚙️  Setting up optimizer...")
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  
    optimizer.add_param_group({"params": g[2], "weight_decay": 0.0}) 
    
    warmup_lr = LinearWarmup(config)

    # Training parameters
    adjustlr_schedule = config['adjustlr_schedule']
    acc_grad = config['acc_grad'] 
    max_epoch = config['max_epoch'] 
    lr_decay = config['lr_decay']
    save_folder = config['save_folder']
    
    torch.backends.cudnn.benchmark = True
    cur_epoch = 1
    loss_acc = 0.0
    ema = EMA(model)
    
    # Training metrics tracking
    training_metrics = {
        'epoch_losses': [],
        'learning_rates': [],
        'training_times': []
    }

    print("🎬 Starting training loop...")
    print("=" * 60)
    
    while(cur_epoch <= max_epoch):
        epoch_start_time = time.time()
        cnt_pram_update = 0
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"📅 Epoch {cur_epoch}/{max_epoch}")
        
        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader): 
            batch_size = batch_clip.shape[0]
            batch_clip = batch_clip.to("cuda")
            
            for idx in range(batch_size):
                batch_bboxes[idx] = batch_bboxes[idx].to("cuda")
                batch_labels[idx] = batch_labels[idx].to("cuda")

            outputs = model(batch_clip)

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

            loss = criterion(outputs, targets) / acc_grad
            loss_acc += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            loss.backward()

            if (iteration + 1) % acc_grad == 0:
                cnt_pram_update = cnt_pram_update + 1
                if cur_epoch == 1:
                    warmup_lr(optimizer, cnt_pram_update)
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

                current_lr = optimizer.param_groups[0]['lr']
                print(f"📈 Epoch: {cur_epoch}, Update: {cnt_pram_update}, Loss: {loss_acc:.6f}, LR: {current_lr:.8f}", flush=True)
                
                # Log to file
                with open(os.path.join(config['save_folder'], "training_log.txt"), "a") as f:
                    f.write(f"Epoch: {cur_epoch}, Update: {cnt_pram_update}, Loss: {loss_acc:.6f}, LR: {current_lr:.8f}\n")

                loss_acc = 0.0

        # End of epoch processing
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Update metrics
        training_metrics['epoch_losses'].append(avg_epoch_loss)
        training_metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        training_metrics['training_times'].append(epoch_time)
        
        print(f"⏱️  Epoch {cur_epoch} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.6f}")
        
        # Learning rate scheduling
        if cur_epoch in adjustlr_schedule:
            old_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups: 
                param_group['lr'] *= lr_decay
            new_lr = optimizer.param_groups[0]['lr']
            print(f"📉 Learning rate adjusted: {old_lr:.8f} -> {new_lr:.8f}")
        
        # Save model checkpoints
        save_path_ema = os.path.join(save_folder, f"level0_ema_epoch_{cur_epoch}.pth")
        torch.save(ema.ema.state_dict(), save_path_ema)

        save_path = os.path.join(save_folder, f"level0_epoch_{cur_epoch}.pth")
        torch.save(model.state_dict(), save_path)

        print(f"💾 Model saved: {save_path}")
        print("-" * 60)

        cur_epoch += 1
    
    # Training completed
    total_time = sum(training_metrics['training_times'])
    print("🎉 Training completed!")
    print(f"⏱️  Total training time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"📊 Final loss: {training_metrics['epoch_losses'][-1]:.6f}")
    print(f"💾 Models saved in: {save_folder}")
    
    # Save training metrics
    metrics_file = os.path.join(save_folder, "training_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("YOLOv12 Level0 Training Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total epochs: {max_epoch}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Average time per epoch: {total_time/max_epoch:.2f}s\n")
        f.write(f"Final loss: {training_metrics['epoch_losses'][-1]:.6f}\n")
        f.write("\nEpoch-wise losses:\n")
        for i, loss in enumerate(training_metrics['epoch_losses'], 1):
            f.write(f"Epoch {i}: {loss:.6f}\n")
    
    print(f"📈 Training metrics saved to: {metrics_file}")


def main():
    """
    Main function to handle command line arguments and start training
    """
    parser = argparse.ArgumentParser(description='YOLOv12 Level0 Training Script')
    parser.add_argument('--config', type=str, default='config/yolov12_level0.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Build configuration with the specified config file
    config = build_config(args.config)
    
    # Verify Level0 configuration
    if config['backbone2D'] != 'yolov12_level0':
        print("⚠️  Warning: backbone2D is not set to 'yolov12_level0'")
        print(f"Current backbone2D: {config['backbone2D']}")
        print("Please check your configuration file.")
    
    print("🔧 Configuration loaded successfully")
    print(f"📄 Config file: {config['config_path']}")
    print(f"🎯 Backbone 2D: {config['backbone2D']}")
    print(f"🎯 Backbone 3D: {config['backbone3D']}")
    print(f"🔗 Fusion module: {config['fusion_module']}")
    
    # Start training
    train_level0_model(config)


if __name__ == "__main__":
    main()