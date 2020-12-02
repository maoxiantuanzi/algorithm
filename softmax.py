"""
@File    :   softmax.py    
@Contact :   pengtt0119@gmail.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/12/2 10:19 上午   ttpeng      1.0         None
"""
import numpy as np


def softmax(x, axis=1):
    """
    对输入x的每一行计算softmax。

    该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。

    代码利用softmax函数的性质: softmax(x) = softmax(x + c)

    参数:
    x -- 一个N维向量，或者M x N维numpy矩阵.
    axis -- 计算的维度，默认按行计算，求每行的最大值

    返回值:
    s -- 经过softmax变换后的结果
    """
    if len(x.shape) <= 1:
        x = x.reshape(1, -1)
    # 计算每行的最大值
    x_max = x.max(axis=axis).reshape((-1, 1))
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x -= x_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


if __name__ == '__main__':
    # test case 1
    test_x1 = np.array([1, 2, 3, 4])
    print('input 1: %s' % test_x1)
    test_softmax1 = softmax(test_x1)
    print('result 1: %s' % test_softmax1)
    # test case 2
    test_x2 = np.array([[1, 2, 4, 1], [5, 6, 5, 7]])
    print('input 2: %s' % test_x2)
    test_softmax2 = softmax(test_x2)
    print('result 2: %s' % test_softmax2)
