import torch
import torch.nn.functional as F


class MnistNN(torch.nn.Module):
    """
    构建模型参考"https://blog.csdn.net/weixin_45891612/article/details/129660091"
    """

    def __init__(self):
        super(MnistNN, self).__init__()
        # 隐藏层1：784*128
        self.hidden1 = torch.nn.Linear(784, 128)
        # 隐藏层2：128*256
        self.hidden2 = torch.nn.Linear(128, 256)
        # 激活层1: relu
        self.activation1 = torch.nn.ReLU()
        # 激活层2: relu
        self.activation2 = torch.nn.ReLU()
        # dropout层1
        self.dropout1 = torch.nn.Dropout(0.01)
        # dropout层2
        self.dropout2 = torch.nn.Dropout(0.01)
        # 激活层3: 256*10
        self.hidden3 = torch.nn.Linear(256, 10)
        # 输出层
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.dropout1(self.activation1(self.hidden1(x)))
        x = self.dropout2(self.activation2(self.hidden2(x)))
        x = self.hidden3(x)
        x = self.softmax(x)
        return x

    def save_weights(self):
        return {
            "hidden1": self.hidden1.weight.detach(),
            "hidden2": self.hidden2.weight.detach(),
            "hidden3": self.hidden3.weight.detach()
        }

    def load_weights(self, weights):
        if weights.get("hidden1") is not None:
            self.hidden1.weight = torch.nn.Parameter(weights.get("hidden1"))
        if weights.get("hidden2") is not None:
            self.hidden2.weight = torch.nn.Parameter(weights.get("hidden2"))
        if weights.get("hidden3") is not None:
            self.hidden3.weight = torch.nn.Parameter(weights.get("hidden3"))

        return self


class LeNet5(torch.nn.Module):
    """
    构建模型参考"10.1109/5.726791"
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        # 定义LeNet-5的层
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.hidden1 = torch.nn.Linear(16 * 5 * 5, 120)  # 输入图像大小为32x32
        self.hidden2 = torch.nn.Linear(120, 84)
        self.hidden3 = torch.nn.Linear(84, 10)  # 10个输出类别

    def forward(self, x):
        x = x.reshape(x.shape[0], 28, 28).unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 展平层
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x

    def save_weights(self):
        return {
            "conv1": self.conv1.weight.detach(),
            "conv2": self.conv2.weight.detach(),
            "hidden1": self.hidden1.weight.detach(),
            "hidden2": self.hidden2.weight.detach(),
            "hidden3": self.hidden3.weight.detach()
        }

    def load_weights(self, weights):
        if weights.get("conv1") is not None:
            self.conv1.weight = torch.nn.Parameter(weights.get("conv1"))
        if weights.get("conv2") is not None:
            self.conv2.weight = torch.nn.Parameter(weights.get("conv2"))
        if weights.get("hidden1") is not None:
            self.hidden1.weight = torch.nn.Parameter(weights.get("hidden1"))
        if weights.get("hidden2") is not None:
            self.hidden2.weight = torch.nn.Parameter(weights.get("hidden2"))
        if weights.get("hidden3") is not None:
            self.hidden3.weight = torch.nn.Parameter(weights.get("hidden3"))

        return self
