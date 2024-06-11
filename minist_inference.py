import random
import os
import argparse

from model import *
from dataloader import get_dataset


def show_image(image, save_path=None):
    """
    图片可视化
    """
    import matplotlib.pyplot as plt

    plt.imshow(image.reshape((28, 28)), cmap="gray")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def predict(model_path=None, model_name="DNN", figure_path=None):
    # 加载推理数据
    test_dataset = get_dataset(False)
    sample_id = random.randint(0, len(test_dataset))
    image, label = test_dataset[sample_id]

    # 加载模型
    assert os.path.isfile(model_path), "no model can find, please check your model path: <{}>".format(model_path)
    if model_name == "LeNet5":
        model = LeNet5().load_weights(torch.load(model_path)).cuda()
    else:
        model = MnistNN().load_weights(torch.load(model_path)).cuda()

    # 推理
    predict_label = torch.argmax(model(image.cuda().reshape(image.shape[0], -1)), dim=1)

    # 展示推理结果
    print("predict label: {}, real label: {}".format(predict_label[0], label))
    show_image(image, figure_path)


if __name__ == '__main__':
    # 推理执行指令: python mnist_inference.py --save-path ${MODEL_PATH}
    # note: best_model.pt为已训练的checkpoint, 准确率96%
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default=None, help="模型持久化路径")
    parser.add_argument("--figure-path", type=str, default=None, help="预测数据存储路径")
    parser.add_argument("--model-name", choices=("DNN", "LeNet5"), default="DNN", help="模型类型: DNN/LeNet5")
    args = parser.parse_args()

    predict(args.save_path, args.model_name, args.figure_path)
