#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOWOv3 多GPU分布式训练模块

这个模块提供了完整的多GPU分布式训练功能，包括：
- DDP (DistributedDataParallel) 支持
- 自动GPU检测和配置
- 确定性训练环境设置
- 进程同步和错误处理
- 训练状态监控和日志记录

使用方法：
    # 直接运行（自动检测GPU数量）
    python scripts/train_multi_gpu.py --config config/yolov12_level0.yaml

    # 指定GPU数量
    python scripts/train_multi_gpu.py --config config/yolov12_level0.yaml --gpus 2

    # 指定特定GPU设备
    python scripts/train_multi_gpu.py --config config/yolov12_level0.yaml --devices 0,1

Author: YOWOv3 Multi-GPU Training Module
Date: 2025.1.15
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import time
import logging
from pathlib import Path

# 添加项目路径 - 兼容Linux和Windows
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保当前工作目录也在路径中
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入项目模块
from utils.build_config import build_config
from scripts.train import train_model
from model.TSN.YOWOv3 import build_yowov3
from cus_datasets.build_dataset import build_dataset
from cus_datasets.collate_fn import collate_fn
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
from utils.EMA import EMA

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_deterministic_training(rank, seed=42):
    """
    设置确定性训练环境

    Args:
        rank (int): 当前进程的rank
        seed (int): 随机种子
    """
    # 设置统一的随机种子（所有进程使用相同种子）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 设置确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置CUDA设备
    torch.cuda.set_device(rank)

    logger.info(f"进程 {rank}: 设置统一随机种子为 {seed}")


def setup_ddp(rank, world_size, master_port='12355'):
    """
    初始化DDP进程组

    Args:
        rank (int): 当前进程的rank
        world_size (int): 总进程数
        master_port (str): 主进程端口
    """
    import socket

    def find_free_port(start_port=12355, max_attempts=100):
        """查找可用端口"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return str(port)
            except OSError:
                continue
        raise RuntimeError(f"无法找到可用端口，尝试范围: {start_port}-{start_port + max_attempts}")

    # 使用临时文件同步端口信息
    import tempfile
    temp_dir = tempfile.gettempdir()
    # 使用固定的标识符而不是PID，确保所有进程使用相同的文件路径
    import time
    session_id = os.environ.get('DDP_SESSION_ID', str(int(time.time())))
    os.environ['DDP_SESSION_ID'] = session_id
    port_file = os.path.join(temp_dir, f'ddp_port_{session_id}_{world_size}.txt')

    if rank == 0:
        try:
            # 尝试使用指定端口
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', int(master_port)))
            actual_port = master_port
        except OSError:
            # 端口被占用，查找可用端口
            actual_port = find_free_port(int(master_port))
            logger.warning(f"端口 {master_port} 被占用，使用端口 {actual_port}")

        # 将端口信息写入文件和环境变量
        try:
            with open(port_file, 'w') as f:
                f.write(actual_port)
                f.flush()  # 确保立即写入磁盘
            # 同时设置环境变量作为备用
            os.environ[f'DDP_PORT_{session_id}_{world_size}'] = actual_port
            logger.info(f"进程 {rank}: 端口信息已写入 {port_file} 和环境变量")
            # 给其他进程一些时间来检测文件
            time.sleep(2)
        except Exception as e:
            logger.warning(f"无法写入端口文件: {e}，使用默认端口")
            actual_port = master_port
    else:
        # 非rank 0进程从文件或环境变量读取端口
        actual_port = master_port
        env_key = f'DDP_PORT_{session_id}_{world_size}'
        max_wait = 60  # 增加等待时间到60秒
        wait_time = 0

        while wait_time < max_wait:
            # 首先尝试从环境变量读取
            if env_key in os.environ:
                actual_port = os.environ[env_key]
                logger.info(f"进程 {rank}: 从环境变量读取端口 {actual_port}")
                break

            # 然后尝试从文件读取
            try:
                if os.path.exists(port_file):
                    with open(port_file, 'r') as f:
                        actual_port = f.read().strip()
                    logger.info(f"进程 {rank}: 从文件读取端口 {actual_port}")
                    break
                else:
                    logger.debug(f"进程 {rank}: 端口文件 {port_file} 不存在，继续等待...")
            except Exception as e:
                logger.warning(f"读取端口文件失败: {e}")

            time.sleep(1.0)  # 增加等待间隔
            wait_time += 1.0

        if wait_time >= max_wait:
            logger.error(f"进程 {rank}: 等待端口信息超时，这将导致端口冲突！")
            logger.error(f"进程 {rank}: 端口文件路径: {port_file}")
            logger.error(f"进程 {rank}: 环境变量键: {env_key}")
            logger.error(f"进程 {rank}: 强制退出以避免端口冲突")
            raise RuntimeError(f"进程 {rank} 无法获取正确的端口信息")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = actual_port

    # 添加超时设置
    timeout = 300  # 5分钟超时

    try:
        # 优先尝试nccl后端
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout if hasattr(torch.distributed, 'default_pg_timeout') else None
        )
        logger.info(f"进程 {rank}: 使用 nccl 后端初始化DDP，端口: {actual_port}")
    except Exception as e:
        logger.warning(f"进程 {rank}: nccl后端初始化失败，尝试gloo后端: {e}")
        try:
            dist.init_process_group(
                backend='gloo',
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.default_pg_timeout if hasattr(torch.distributed,
                                                                        'default_pg_timeout') else None
            )
            logger.info(f"进程 {rank}: 使用 gloo 后端初始化DDP，端口: {actual_port}")
        except Exception as e2:
            logger.error(f"进程 {rank}: DDP初始化完全失败: {e2}")
            logger.error(f"尝试的端口: {actual_port}")
            logger.error("建议解决方案:")
            logger.error("1. 检查是否有其他训练进程在运行: ps aux | grep python")
            logger.error("2. 杀死占用端口的进程: sudo lsof -ti:12355 | xargs kill -9")
            logger.error("3. 使用不同的端口: --master-port 12356")
            raise e2

    logger.info(f"进程 {rank}: DDP初始化完成")


def cleanup_ddp():
    """清理DDP进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_worker(rank, world_size, config, args):
    """
    DDP训练工作进程

    Args:
        rank (int): 当前进程的rank
        world_size (int): 总进程数
        config (dict): 训练配置
        args: 命令行参数
    """
    try:
        logger.info(f"\n=== 进程 {rank} 开始多GPU训练 ===")

        # 设置确定性训练环境
        setup_deterministic_training(rank, seed=42)

        # 设置设备
        device = torch.device(f'cuda:{rank}')
        logger.info(f"进程 {rank}: 使用设备 {device}")

        # 初始化DDP
        setup_ddp(rank, world_size)

        # 同步所有进程
        logger.info(f"进程 {rank}: 等待所有进程同步...")
        dist.barrier()

        # 创建模型
        logger.info(f"进程 {rank}: 创建模型")
        model = build_yowov3(config)
        model = model.to(device)

        # 打印模型信息（仅在rank 0）
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"模型参数总数: {total_params:,}")

            # 打印前几个参数的详细信息
            param_count = 0
            for name, param in model.named_parameters():
                if param_count < 5:
                    logger.info(f"参数 {name} - 形状: {param.shape}, 步长: {param.stride()}")
                    param_count += 1
                else:
                    break

        # 同步模型创建
        dist.barrier()

        # 包装为DDP模型
        logger.info(f"进程 {rank}: 包装DDP模型")
        ddp_model = DDP(
            model,
            device_ids=[rank],
            find_unused_parameters=True,  # 处理可能的未使用参数
            broadcast_buffers=False  # 避免buffer广播问题
        )

        # 创建数据集和数据加载器
        logger.info(f"进程 {rank}: 创建数据集")
        dataset = build_dataset(config, phase='train')

        # 创建分布式采样器
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        # 调整批次大小（每个GPU的批次大小）
        per_gpu_batch_size = config['batch_size'] // world_size
        if per_gpu_batch_size == 0:
            per_gpu_batch_size = 1
            logger.warning(f"批次大小太小，调整为每GPU 1个样本")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_gpu_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 4) // world_size,
            pin_memory=True,
            drop_last=True
        )

        logger.info(f"进程 {rank}: 每GPU批次大小: {per_gpu_batch_size}")

        # 创建优化器和损失函数
        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=config.get('lr', 0.0001),
            weight_decay=config.get('weight_decay', 0.0005)
        )

        criterion = build_loss(ddp_model, config)

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('max_epoch', 10)
        )

        # EMA（仅在rank 0）
        ema = None
        if rank == 0 and config.get('use_ema', True):
            ema = EMA(ddp_model, decay=config.get('ema_decay', 0.9999))

        # 训练循环
        logger.info(f"进程 {rank}: 开始训练循环")

        for epoch in range(config.get('max_epoch', 10)):
            # 设置采样器的epoch
            sampler.set_epoch(epoch)

            # 训练一个epoch
            ddp_model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (clips, boxes, labels) in enumerate(dataloader):
                # 数据移到设备
                clips = clips.to(device, non_blocking=True)
                boxes = [box.to(device, non_blocking=True) for box in boxes]
                labels = [label.to(device, non_blocking=True) for label in labels]

                # 组合targets - 与train.py格式保持一致
                targets = []
                for i, (bboxes, labels) in enumerate(zip(boxes, labels)):
                    nbox = bboxes.shape[0]
                    nclass = labels.shape[1]
                    target = torch.zeros(nbox, 5 + nclass, device=device)
                    target[:, 0] = i
                    target[:, 1:5] = bboxes
                    target[:, 5:] = labels
                    targets.append(target)

                targets = torch.cat(targets, dim=0)

                # 前向传播
                optimizer.zero_grad()
                outputs = ddp_model(clips)

                # 计算损失
                loss = criterion(outputs, targets)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

                # 优化器步进
                optimizer.step()

                # 更新EMA
                if ema is not None:
                    ema.update(ddp_model)

                epoch_loss += loss.item()
                num_batches += 1

                # 打印进度（仅rank 0）
                if rank == 0 and batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{config.get('max_epoch', 10)}, "
                                f"Batch {batch_idx}/{len(dataloader)}, "
                                f"Loss: {loss.item():.4f}")

            # 同步所有进程
            dist.barrier()

            # 计算平均损失
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            # 学习率调度
            scheduler.step()

            # 保存检查点（仅rank 0）
            if rank == 0:
                logger.info(f"Epoch {epoch + 1} 完成, 平均损失: {avg_loss:.4f}")

                # 保存模型
                if (epoch + 1) % config.get('save_interval', 5) == 0:
                    save_path = os.path.join(
                        config['save_folder'],
                        f'checkpoint_epoch_{epoch + 1}.pth'
                    )

                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': ddp_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': avg_loss,
                    }

                    if ema is not None:
                        checkpoint['ema_state_dict'] = ema.ema.state_dict()

                    torch.save(checkpoint, save_path)
                    logger.info(f"保存检查点: {save_path}")

        logger.info(f"进程 {rank}: 训练完成")

    except Exception as e:
        logger.error(f"进程 {rank} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise e

    finally:
        # 清理分布式进程组
        if dist.is_initialized():
            dist.destroy_process_group()

        # 清理端口文件和环境变量（只有rank 0负责清理）
        if rank == 0:
            try:
                import tempfile
                temp_dir = tempfile.gettempdir()
                port_file = os.path.join(temp_dir, f'ddp_port_{os.getpid()}_{world_size}.txt')
                if os.path.exists(port_file):
                    os.remove(port_file)
                    logger.info(f"已清理端口文件: {port_file}")

                # 清理环境变量
                session_id = os.environ.get('DDP_SESSION_ID')
                if session_id:
                    env_key = f'DDP_PORT_{session_id}_{world_size}'
                    if env_key in os.environ:
                        del os.environ[env_key]
                        logger.info(f"已清理环境变量: {env_key}")
                    # 清理session_id环境变量
                    if 'DDP_SESSION_ID' in os.environ:
                        del os.environ['DDP_SESSION_ID']
            except Exception as e:
                logger.warning(f"清理端口信息失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOWOv3 Multi-GPU Training')
    parser.add_argument('--config', type=str, default='config/yolov12_level0.yaml',
                        help='配置文件路径')
    parser.add_argument('--gpus', type=int, default=None,
                        help='使用的GPU数量（默认自动检测）')
    parser.add_argument('--devices', type=str, default=None,
                        help='指定GPU设备ID，用逗号分隔（如：0,1,2）')
    parser.add_argument('--master-port', type=str, default='12355',
                        help='DDP主进程端口')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 清理可能存在的分布式进程组
    if dist.is_initialized():
        logger.info("检测到已存在的分布式进程组，正在清理...")
        dist.destroy_process_group()
        time.sleep(2)  # 等待清理完成

    args = parser.parse_args()

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，无法进行多GPU训练")
        logger.info("解决方案:")
        logger.info(
            "1. 如果要在CPU上训练，请使用: python scripts/train.py --config config/yolov12n_optimized_fixed.yaml")
        logger.info("2. 如果要使用GPU，请确保:")
        logger.info("   - 安装了CUDA兼容的PyTorch版本")
        logger.info("   - GPU驱动程序正确安装")
        logger.info("   - 检查命令: python -c 'import torch; print(torch.cuda.is_available())'")
        return

    # 确定使用的GPU设备
    if args.devices:
        # 用户指定设备
        device_ids = [int(x.strip()) for x in args.devices.split(',')]
        world_size = len(device_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
        logger.info(f"使用指定GPU设备: {device_ids}")
    else:
        # 自动检测或用户指定数量
        available_gpus = torch.cuda.device_count()
        world_size = args.gpus if args.gpus else available_gpus
        world_size = min(world_size, available_gpus)
        logger.info(f"检测到 {available_gpus} 个GPU，使用 {world_size} 个GPU")

    if world_size < 2:
        logger.warning("检测到只有1个GPU，将使用单GPU训练模式")
        logger.info("建议使用: python scripts/train.py --config config/yolov12n_optimized_fixed.yaml")
        logger.info("如果要强制使用多GPU训练，请确保有至少2个GPU可用")

        # 回退到单GPU训练
        try:
            from scripts.train import train_model
            config_path = Path(args.config)
            config = build_config(str(config_path))
            logger.info("开始单GPU训练...")
            train_model(config)
            logger.info("✅ 单GPU训练完成！")
        except Exception as e:
            logger.error(f"❌ 单GPU训练失败: {e}")
            import traceback
            traceback.print_exc()
        return

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return

    config = build_config(str(config_path))

    # 创建保存目录
    save_folder = config.get('save_folder', 'runs/train_multi_gpu')
    os.makedirs(save_folder, exist_ok=True)
    config['save_folder'] = save_folder

    logger.info(f"\n{'=' * 60}")
    logger.info(f" YOWOv3 多GPU分布式训练")
    logger.info(f"{'=' * 60}")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"GPU数量: {world_size}")
    logger.info(f"批次大小: {config['batch_size']} (总计)")
    logger.info(f"训练轮数: {config.get('max_epoch', 10)}")
    logger.info(f"保存目录: {save_folder}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"{'=' * 60}\n")

    # 启动多进程训练
    try:
        # 清理可能存在的旧端口文件和环境变量
        import tempfile
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if file.startswith(f'ddp_port_') and file.endswith(f'_{world_size}.txt'):
                try:
                    os.remove(os.path.join(temp_dir, file))
                    logger.info(f"清理旧端口文件: {file}")
                except Exception as e:
                    logger.warning(f"清理旧端口文件失败: {e}")

        # 清理可能存在的旧环境变量
        for key in list(os.environ.keys()):
            if key.startswith('DDP_PORT_') and key.endswith(f'_{world_size}'):
                del os.environ[key]
                logger.info(f"清理旧环境变量: {key}")
        if 'DDP_SESSION_ID' in os.environ:
            del os.environ['DDP_SESSION_ID']

        mp.spawn(
            train_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True
        )
        logger.info("\n✅ 多GPU训练成功完成！")
    except Exception as e:
        logger.error(f"\n❌ 多GPU训练失败: {e}")
        import traceback
        traceback.print_exc()


def main_with_config(config, args):
    """
    从main.py调用的多GPU训练入口函数

    Args:
        config (dict): 训练配置
        args: 命令行参数对象
    """
    # 清理可能存在的分布式进程组
    if dist.is_initialized():
        logger.info("检测到已存在的分布式进程组，正在清理...")
        dist.destroy_process_group()
        time.sleep(2)  # 等待清理完成

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，无法进行多GPU训练")
        return

    # 确定使用的GPU设备
    if args.devices:
        # 用户指定设备
        device_ids = [int(x.strip()) for x in args.devices.split(',')]
        world_size = len(device_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
        logger.info(f"使用指定GPU设备: {device_ids}")
    else:
        # 自动检测或用户指定数量
        available_gpus = torch.cuda.device_count()
        world_size = args.gpus if args.gpus else available_gpus
        world_size = min(world_size, available_gpus)
        logger.info(f"检测到 {available_gpus} 个GPU，使用 {world_size} 个GPU")

    if world_size < 2:
        logger.warning("检测到只有1个GPU，将使用单GPU训练模式")
        logger.info("建议使用: python main.py --mode train --config config/yolov12n_optimized_fixed.yaml")

        # 回退到单GPU训练
        try:
            from scripts.train import train_model
            logger.info("开始单GPU训练...")
            train_model(config)
            logger.info("✅ 单GPU训练完成！")
        except Exception as e:
            logger.error(f"❌ 单GPU训练失败: {e}")
            import traceback
            traceback.print_exc()
        return

    # 创建保存目录
    save_folder = config.get('save_folder', 'runs/train_multi_gpu')
    os.makedirs(save_folder, exist_ok=True)
    config['save_folder'] = save_folder

    logger.info(f"\n{'=' * 60}")
    logger.info(f" YOWOv3 多GPU分布式训练")
    logger.info(f"{'=' * 60}")
    logger.info(f"GPU数量: {world_size}")
    logger.info(f"批次大小: {config['batch_size']} (总计)")
    logger.info(f"训练轮数: {config.get('max_epoch', 10)}")
    logger.info(f"保存目录: {save_folder}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"{'=' * 60}\n")

    # 启动多进程训练
    try:
        # 清理可能存在的旧端口文件和环境变量
        import tempfile
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if file.startswith(f'ddp_port_') and file.endswith(f'_{world_size}.txt'):
                try:
                    os.remove(os.path.join(temp_dir, file))
                    logger.info(f"清理旧端口文件: {file}")
                except Exception as e:
                    logger.warning(f"清理旧端口文件失败: {e}")

        # 清理可能存在的旧环境变量
        for key in list(os.environ.keys()):
            if key.startswith('DDP_PORT_') and key.endswith(f'_{world_size}'):
                del os.environ[key]
                logger.info(f"清理旧环境变量: {key}")
        if 'DDP_SESSION_ID' in os.environ:
            del os.environ['DDP_SESSION_ID']

        mp.spawn(
            train_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True
        )
        logger.info("\n✅ 多GPU训练成功完成！")
    except Exception as e:
        logger.error(f"\n❌ 多GPU训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()