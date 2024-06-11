#### 简单DNN例子
  - 模型训练步骤
    1. 定义训练数据, 包括学习率、模型层数、隐层大小等超参数
    2. 加载训练集、验证集
    3. 初始化模型
    4. 初始化优化器状态, 用于跟踪和更新需要学习的模型参数
    5. 设置损失函数
    6. 开始训练: forward -> backward -> 优化器更新参数 -> 验证集测试当前模型泛化性(可选) -> checkpoint(可选)
  - 模型推理步骤
    1. 加载推理数据
    2. 加载训练好的模型
    3. 推理获得结果
    4. 反馈预测结果
  - 用例演示
    - 数据开源: torch社区mnist数据
    - 数据分布: 训练集数据60000例, 测试集数据10000例
    - DNN模型: linear(784\*128) -> relu -> dropout(0.01) -> linear(128\*256) -> relu -> dropout(0.01) -> linear(256\*10) -> softmax
    - 测试环境: cuda: 12.1, pytorch: 2.1.2, python: 3.11.4
    - 推理模型: best_model.pt(准确率: 96%)
    - 训练指令
      ```shell
        python mnist_train.py --save-path ${MODEL_PATH}
      ```
    - 单机数据并行训练指令
      ```shell
        torchrun --standalone --nproc_per_node ${DP_SIZE} mnist_train.py --save-path ${MODEL_PATH} --dp
      ```
    - 推理指令
      ```shell
        python mnist_inference.py --save-path ${MODEL_PATH}
      ```
