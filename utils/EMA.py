import copy
import math
import torch


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA - 确保使用原始模型而不是DDP包装的模型
        if hasattr(model, 'module'):
            self.ema = copy.deepcopy(model.module).eval()  # DDP模型，使用内部模型
        else:
            self.ema = copy.deepcopy(model).eval()  # 普通模型

        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            # 获取模型的state_dict，处理DDP包装的情况
            if hasattr(model, 'module'):
                msd = model.module.state_dict()  # DDP模型，获取内部模型的state_dict
            else:
                msd = model.state_dict()  # 普通模型

            # 更新EMA参数
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    if k in msd:
                        v *= d
                        v += (1 - d) * msd[k].detach()
                    else:
                        # 如果键不存在，记录警告但不中断训练
                        print(f"Warning: EMA parameter '{k}' not found in model state_dict")