#!/usr/bin/env python3
import argparse
import torch
from utils.build_config import build_config
from scripts import train, ucf_eval, ava_eval, detect, live, onnx
from scripts import train_multi_gpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOWOv3")

    parser.add_argument('-m', '--mode', type=str, help='train/eval/live/detect/onnx/fusion_mamba')
    parser.add_argument('-cf', '--config', type=str, help='path to config file')
    parser.add_argument('--name', type=str, default=None, help='experiment name (for fusion_mamba mode)')
    parser.add_argument('--stage', type=str, choices=['1', '2', '3', '4', 'all'],
                        default='all', help='training stage (for fusion_mamba mode)')

    # 多GPU训练相关参数
    parser.add_argument('--multi-gpu', action='store_true', help='启用多GPU分布式训练')
    parser.add_argument('--gpus', type=int, default=None, help='使用的GPU数量（默认自动检测）')
    parser.add_argument('--devices', type=str, default=None, help='指定GPU设备ID，用逗号分隔（如：0,1,2）')
    parser.add_argument('--master-port', type=str, default='12355', help='DDP主进程端口')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    config = build_config(args.config)

    if args.mode == 'train':
        # 判断是否使用多GPU训练
        if args.multi_gpu:
            # 检查CUDA可用性
            if not torch.cuda.is_available():
                print("错误: CUDA不可用，无法进行多GPU训练")
                print("解决方案:")
                print("1. 如果要在CPU上训练，请移除 --multi-gpu 参数")
                print("2. 如果要使用GPU，请确保安装了CUDA兼容的PyTorch版本")
                exit(1)

            # 确定使用的GPU数量
            available_gpus = torch.cuda.device_count()
            if args.devices:
                device_ids = [int(x.strip()) for x in args.devices.split(',')]
                world_size = len(device_ids)
            else:
                world_size = args.gpus if args.gpus else available_gpus
                world_size = min(world_size, available_gpus)

            if world_size < 2:
                print(f"警告: 检测到只有{world_size}个GPU，将使用单GPU训练模式")
                train.train_model(config=config)
            else:
                print(f"启动多GPU训练，使用{world_size}个GPU")
                # 调用多GPU训练模块
                train_multi_gpu.main_with_config(config, args)
        else:
            # 单GPU训练
            train.train_model(config=config)

    elif args.mode == 'eval':
        if config['dataset'] == 'ucf' or config['dataset'] == 'jhmdb' or config['dataset'] == 'ucfcrime':
            ucf_eval.eval(config=config)
        elif config['dataset'] == 'ava':
            ava_eval.eval(config=config)

    elif args.mode == 'detect':
        detect.detect(config=config)

    elif args.mode == 'live':
        live.detect(config=config)

    elif args.mode == 'onnx':
        onnx.export2onnx(config=config)

    elif args.mode == 'fusion_mamba':
        print("错误: fusion_mamba模式暂不可用，fusion_mamba_experiment模块不存在")
        print("可用模式: train, eval, detect, live, onnx")
        exit(1)