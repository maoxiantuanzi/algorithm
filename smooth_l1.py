"""
@File    :   smooth_l1.py    
@Contact :   pengtt0119@gmail.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/30 11:38 上午   ttpeng      1.0         None
"""

import torch


def _smooth_l1_loss(input, target, reduction='none'):
    """
    :type: (Tensor, Tensor) -> Tensor
    """
    t = torch.abs(input - target)
    ret = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def smooth_l1_loss(input, target, beta=1. / 9, reduction='none'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


if __name__ == '__main__':
    t1 = torch.randn(4)
    t2 = torch.randn(4)
    print(t1)
    print(t2)
    res1 = _smooth_l1_loss(t1, t2, reduction='mean')
    res2 = smooth_l1_loss(t1, t2)

    print(res1)
    print(res2)
