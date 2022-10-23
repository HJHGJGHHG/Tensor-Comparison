import torch
import paddle
import numpy as np

from functools import reduce


def analyze(a, threshold=1e-15):
    a = a.detach().cpu().numpy()
    print("---分析矩阵---")
    values_counts = reduce(lambda x, y: x * y, a.shape)  # 元素个数
    abs_value = np.abs(a)
    print("稀疏度:{}".format(np.sum(a < threshold) / values_counts))
    print("1范数:{}".format(np.sum(abs_value)))
    print("F范数:{}".format(np.sqrt(np.sum(abs_value ** 2))))
    print("无穷范数:{}".format(np.max(abs_value)))
    print("谱范数:{}".format(np.linalg.norm(a, ord=2)))
    print("秩:{}".format(np.linalg.matrix_rank(a)))


def compare(a, b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    assert a.shape == b.shape
    abs_dif = np.abs(a - b)
    print("---比较两个矩阵---")
    print("逐点平均误差:{}".format(np.mean(abs_dif)))
    print("逐点最大误差:{}".format(np.max(abs_dif)))
    print("逐点最小误差:{}".format(np.min(abs_dif)))
    print("逐点平方误差和:{}".format(np.sum((a - b) ** 2)))


if __name__ == "__main__":
    # a = torch.Tensor(np.random.rand(4, 4))
    a = torch.Tensor([
        [1, 2],
        [3, 4]
    ])
    # b = torch.Tensor(np.random.rand(4, 4))
    b = torch.Tensor([
        [4, 2],
        [3, 1]
    ])
    compare(a, b)
    analyze(a)
