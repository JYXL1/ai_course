import argparse
import torch.nn.functional as F

from model import MnistNN
from dataloader import get_dataloader
from dp_util import *


def train(model_path=None, dp=False):
    if dp:
        # 数据并行模式建立通信域
        init_group()
    # 定义训练参数
    train_batch_size, test_batch_size = 32, 64
    lr = 0.001
    epochs = 10 # 10epochs 基本达到收敛
    # 加载训练集, 测试集
    train_loader, test_loader = get_dataloader(train_batch_size, test_batch_size)
    if not dp or (dp and is_first_rank()):
        print("train data size: {}, test data size: {}".format(len(train_loader.dataset), len(test_loader.dataset)), flush=True)
    # 初始化模型
    model = MnistNN().cuda()
    # 设置优化器
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 定义损失函数
    loss_func = F.cross_entropy
    best_model, best_acc = None, 0

    # 开始训练
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            opt.zero_grad()
            # 前向计算损失
            data = bcast_and_slice(images.cuda().reshape(labels.shape[0], -1), dp)
            out = model(data)
            loss = loss_func(out, F.one_hot(bcast_and_slice(labels.cuda(), dp), 10).float())
            # 反向计算梯度
            loss.backward()
            if dp:
                # 数据并行模式需要在更新参数前聚合梯度
                all_reduce_grads([param.grad.data for param in model.parameters()])
            # 优化器更新参数
            opt.step()
        if not dp or (dp and is_first_rank()):
            model.eval()
            # 测试集验证当前模型准确性
            with torch.no_grad():
                acc = 0.0
                for images, labels in test_loader:
                    predict_labels = torch.argmax(model(images.cuda().reshape(labels.shape[0], -1)), dim=1)
                    acc += torch.sum(predict_labels==labels.cuda())
                acc /= len(test_loader.dataset)
                if acc > best_acc:
                    best_model, best_acc = model.save_weights(), acc
                print("after {} epoch, test acc: {}, best acc: {}".format(epoch+1, acc, best_acc), flush=True)

    # 持久化最优模型
    if model_path and (not dp or (dp and is_first_rank())):
        torch.save(best_model, model_path)


if __name__ == '__main__':
    # 训练执行指令: python mnist_train.py --save-path ${MODEL_PATH}
    # 单机数据并行训练指令: torchrun --standalone --nproc_per_node ${DP_SIZE} mnist_train.py --save-path ${MODEL_PATH} --dp
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default=None, help="模型持久化路径")
    parser.add_argument("--dp", action="store_true", help="使用数据并行模式")
    args = parser.parse_args()

    train(args.save_path, args.dp)
