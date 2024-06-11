import torch
from torchvision import datasets, transforms


def get_dataset(is_train=True):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化图像
    ])

    return datasets.MNIST(root='./data', train=is_train, download=True, transform=transform)


def get_dataloader(train_batch_size, test_batch_size):
    # 加载MNIST数据集
    train_dataset = get_dataset(True)
    test_dataset = get_dataset(False)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
