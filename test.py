import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from googlenet import SimpleGoogLeNet


# 数据预处理和加载
def load_data(batch_size=64, data_path='D:/workspace/data'):
    """
    加载 MNIST 测试数据集并返回 DataLoader。

    参数:
    - batch_size (int): 每个 batch 的大小，默认为 64。
    - data_path (str): 数据集存储路径，默认为 'D:/workspace/data'。

    返回:
    - DataLoader: 用于遍历 MNIST 测试集的 DataLoader 对象。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 正规化到 [-1, 1] 范围
    ])

    # 加载 MNIST 测试数据集
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


# 加载训练好的模型
def load_model(model_path='googlenet_mnist.pth', num_classes=10):
    """
    加载训练好的 SimpleGoogLeNet 模型，并将其移到可用的设备（CPU 或 GPU）。

    参数:
    - model_path (str): 训练好的模型文件路径，默认为 'googlenet_mnist.pth'。
    - num_classes (int): 模型输出的类别数，默认为 10（适用于 MNIST 数据集）。

    返回:
    - model (torch.nn.Module): 加载后的模型。
    - device (torch.device): 当前模型所在的设备（CPU 或 GPU）。
    """
    model = SimpleGoogLeNet(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, device


# 测试模型并获取准确率
def evaluate_model(model, device, test_loader):
    """
    评估模型准确率，并获取前六张图片的预测结果。

    参数:
    - model (torch.nn.Module): 训练好的模型。
    - device (torch.device): 当前模型所在的设备（CPU 或 GPU）。
    - test_loader (DataLoader): 用于加载测试数据的 DataLoader。

    返回:
    - accuracy (float): 测试集上的准确率。
    - all_images (list): 存储前六张测试图片的列表。
    - all_labels (list): 存储前六张测试图片的真实标签列表。
    - all_predictions (list): 存储前六张测试图片的预测标签列表。
    """
    correct = 0
    total = 0
    all_images = []  # 用于存储前六张图片
    all_labels = []  # 用于存储前六张图片的标签
    all_predictions = []  # 用于存储前六张图片的预测标签

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 存储前六张图片的输入、标签和预测
            if len(all_images) < 6:
                all_images.extend(inputs.cpu().numpy()[:6])
                all_labels.extend(labels.cpu().numpy()[:6])
                all_predictions.extend(predicted.cpu().numpy()[:6])

    # 计算测试集准确率
    accuracy = 100 * correct / total
    return accuracy, all_images, all_labels, all_predictions


# 可视化前六张图片及其预测结果
def visualize_results(all_images, all_labels, all_predictions):
    """
    可视化前六张图片，并显示其真实标签和预测标签。

    参数:
    - all_images (list): 存储前六张图片的列表。
    - all_labels (list): 存储前六张图片的真实标签列表。
    - all_predictions (list): 存储前六张图片的预测标签列表。

    返回:
    - None: 此函数不返回任何内容，直接展示可视化结果。
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # 2行3列
    axes = axes.flatten()  # 展平axes，便于索引

    for i in range(6):
        ax = axes[i]
        img = all_images[i].transpose((1, 2, 0))  # 从 Tensor 格式转换为 HWC 格式
        img = (img * 0.5) + 0.5  # 恢复到 [0, 1] 范围

        ax.imshow(img)  # 显示图片
        ax.set_title(f"True: {all_labels[i]}, Pred: {all_predictions[i]}")  # 显示真实标签和预测标签
        ax.axis('off')  # 不显示坐标轴

    plt.tight_layout()  # 自动调整子图间距
    plt.show()


# 主函数
def main():
    """
    主函数，加载数据、加载模型、评估模型并可视化结果。

    参数:
    - None: 此函数没有输入参数。

    返回:
    - None: 此函数不返回任何内容，直接执行模型评估和可视化。
    """
    # 加载数据
    test_loader = load_data()

    # 加载模型
    model, device = load_model()

    # 评估模型
    accuracy, all_images, all_labels, all_predictions = evaluate_model(model, device, test_loader)

    # 输出测试准确率
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 可视化前六张图片及其预测结果
    visualize_results(all_images, all_labels, all_predictions)


# 执行主函数
if __name__ == "__main__":
    main()
