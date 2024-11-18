import torch
import torch.nn as nn


class SimpleGoogLeNet(nn.Module):
    """
    SimpleGoogLeNet 是一个简化版的 GoogLeNet 模型，专门为处理 MNIST 数据集（灰度图像）设计。
    该模型使用了三个简化的 Inception 模块，后接一个全连接分类器。

    参数:
    - num_classes (int): 输出类别的数量，默认为 10（适用于 MNIST 数据集）。

    主要层次结构：
    1. 三个卷积 + ReLU + MaxPool 层（类似于 Inception 模块）。
    2. 一个全连接层，用于图像分类。
    """
    def __init__(self, num_classes=10):
        super(SimpleGoogLeNet, self).__init__()

        # 第一个 Inception 模块
        self.inception1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道1，输出通道32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 最大池化层，大小为2x2
        )

        # 第二个 Inception 模块
        self.inception2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 输入通道32，输出通道64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 最大池化层，大小为2x2
        )

        # 第三个 Inception 模块
        self.inception3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输入通道64，输出通道128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 最大池化层，大小为2x2
        )

        # 分类器：全连接层 + Dropout 层
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 1024),  # 将特征图展平成一维向量，输入到全连接层
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合，丢弃50%的神经元
            nn.Linear(1024, num_classes)  # 输出层，num_classes 个类别
        )

    def forward(self, x):
        """
        前向传播函数，执行数据流经过网络。

        参数:
        - x (torch.Tensor): 输入数据，形状为 (batch_size, 1, 28, 28)，适用于 MNIST 数据集。

        返回:
        - x (torch.Tensor): 网络的输出，形状为 (batch_size, num_classes)。
        """
        x = self.inception1(x)  # 第一个 Inception 模块
        x = self.inception2(x)  # 第二个 Inception 模块
        x = self.inception3(x)  # 第三个 Inception 模块

        # 展平特征图以进入全连接层
        x = x.view(x.size(0), -1)  # x.size(0) 是 batch_size，-1 自动计算其他维度的大小

        # 分类部分
        x = self.fc(x)  # 输出最终的类别预测

        return x
