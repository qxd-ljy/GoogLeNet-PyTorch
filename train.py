import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from googlenet import SimpleGoogLeNet
from tqdm import tqdm
import torch.multiprocessing as mp


# 数据预处理和加载
def get_data_loader(batch_size=64):
    """
    加载 MNIST 数据集，并返回 DataLoader 用于批量加载数据。

    参数:
    - batch_size (int): 每个 batch 的大小，默认为 64。

    返回:
    - DataLoader: 用于加载 MNIST 训练数据的 DataLoader 对象。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 正规化到 [-1, 1] 范围
    ])

    train_dataset = datasets.MNIST(root='D:/workspace/data', train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)


# 模型初始化
def initialize_model(num_classes=10):
    """
    初始化 SimpleGoogLeNet 模型，并将其移动到可用设备。

    参数:
    - num_classes (int): 模型输出的类别数，默认为 10（适用于 MNIST 数据集）。

    返回:
    - model (torch.nn.Module): 初始化的模型。
    - device (torch.device): 当前模型所在的设备（CPU 或 GPU）。
    """
    model = SimpleGoogLeNet(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


# 训练过程
def train_epoch(model, device, train_loader, criterion, optimizer):
    """
    训练一个 epoch，并返回该 epoch 的平均损失和准确率。

    参数:
    - model (torch.nn.Module): 训练的模型。
    - device (torch.device): 当前模型所在的设备（CPU 或 GPU）。
    - train_loader (DataLoader): 用于加载训练数据的 DataLoader。
    - criterion (torch.nn.Module): 损失函数。
    - optimizer (torch.optim.Optimizer): 优化器。

    返回:
    - epoch_loss (float): 当前 epoch 的平均损失。
    - epoch_accuracy (float): 当前 epoch 的准确率。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc="Training", unit="batch", ncols=100) as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 正常前向和反向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条显示
            pbar.set_postfix(loss=running_loss / (total // 64), accuracy=100 * correct / total)

    return running_loss / len(train_loader), 100 * correct / total


# 保存模型
def save_model(model, filename='googlenet_mnist.pth'):
    """
    保存训练好的模型。

    参数:
    - model (torch.nn.Module): 训练好的模型。
    - filename (str): 模型保存的路径，默认为 'googlenet_mnist.pth'。

    返回:
    - None: 该函数没有返回值。
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")


# 训练完整模型
def train_model(num_epochs=5, batch_size=64, lr=0.001):
    """
    训练整个模型，包括加载数据、初始化模型、训练和保存模型。

    参数:
    - num_epochs (int): 训练的 epoch 数，默认为 5。
    - batch_size (int): 每个 batch 的大小，默认为 64。
    - lr (float): 学习率，默认为 0.001。

    返回:
    - None: 该函数没有返回值，直接进行训练并保存模型。
    """
    # 获取数据加载器
    train_loader = get_data_loader(batch_size)

    # 初始化模型和设备
    model, device = initialize_model()

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器

    # 训练模型
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        epoch_loss, epoch_accuracy = train_epoch(model, device, train_loader, criterion, optimizer)

        # 打印每个 epoch 的平均损失和准确率
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # 保存训练好的模型
    save_model(model)


# 入口函数
if __name__ == '__main__':
    mp.set_start_method('spawn')  # 设置启动方法为 'spawn'，避免 Windows 上的多进程问题
    train_model()  # 运行训练过程
