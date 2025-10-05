import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_data(filename, batchSize=0): # 读取数据，并整理为pytorch需要的格式

    dataset = pd.read_csv(filename, delimiter=',', encoding='utf-8')
    # process the dataset
    attributeMap = {} # 建立空的属性文字到数值的映射字典
    # 色泽: 3个属性值， 分别赋0, 0.5, 1数值
    attributeMap['浅白'] = 0
    attributeMap['青绿'] = 0.5
    attributeMap['乌黑'] = 1
    #根蒂
    attributeMap['蜷缩'] = 0
    attributeMap['稍蜷'] = 0.5
    attributeMap['硬挺'] = 1
    #敲声
    attributeMap['沉闷'] = 0
    attributeMap['浊响'] = 0.5
    attributeMap['清脆'] = 1
    # 纹理
    attributeMap['模糊'] = 0
    attributeMap['稍糊'] = 0.5
    attributeMap['清晰'] = 1
    # 脐部
    attributeMap['凹陷'] = 0
    attributeMap['稍凹'] = 0.5
    attributeMap['平坦'] = 1
    # 触感
    attributeMap['硬滑'] = 0
    attributeMap['软粘'] = 1
    # 好瓜
    attributeMap['否'] = 0
    attributeMap['是'] = 1

    del dataset['编号'] # 删除无用的‘编号’
    dataset = np.array(dataset)
    m, n = np.shape(dataset)

    for i in range(m):
        for j in range(n):
            if dataset[i,j] in attributeMap:
                dataset[i,j] = attributeMap[dataset[i,j]]
                dataset[i,j] = round(dataset[i,j], 3)

    Y = torch.tensor(np.float32(dataset[:, n-1])) # 标签
    if batchSize == 0: batchSize = len(Y)
    X = torch.tensor(np.float32(dataset[:, :n-1])) # 样本的特征矢量
    torch_dataset = torch.utils.data.TensorDataset(X, torch.tensor(Y))

    loader = torch.utils.data.DataLoader(
        dataset = torch_dataset,
        batch_size = batchSize,
        shuffle = False
    )

    return loader

class Model(torch.nn.Module): # 创建BP神经网络

    def __init__(self, input_nodes, hidd_nodes, output_nodes):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(input_nodes, hidd_nodes) # 输入层与隐层之间是线性
        self.linear2 = torch.nn.Linear(hidd_nodes, output_nodes) # 隐层和输出层之间是线性
        self.relu = torch.nn.ReLU() # 通过ReLU实现非线性映射

    def forward(self, x): # 神经网络的前向过程
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x

def learning(ds, epochs=200):
    input_nodes = len(ds.dataset[0][0]) # 输入层节点数
    hidd_nodes = 100 # 隐层节点数
    output_nodes = 2 # 输出层节点数，这里使用One-hot编码

    loss = list()
    acc = list()

    model = Model(input_nodes, hidd_nodes, output_nodes) # 创建BP神经网络实例
    cost = torch.nn.MSELoss(reduction='mean') # 损失计算：采用最小二乘法
    optimizer = torch.optim.Adam(model.parameters()) # 采用Adam进行梯度下降

    for epoch in range(epochs): # 按迭代次数计算每一轮次的损失和准确率
        sum_loss = 0
        train_correct = 0
        for data in ds:
            inputs, labels = data
            outputs = model(inputs)
            optimizer.zero_grad() # 每次计算梯度增量时，先初始化为0
            _loss = cost(outputs, labels.unsqueeze(-1)) # 计算损失
            _loss.backward() # 损失反向传播
            optimizer.step() # 反向传播时按先前Adam进行优化

            _, id = torch.max(outputs.data, 1) # 预测的标签
            sum_loss += float(_loss.data) # 本轮损失累计
            train_correct += torch.sum(id == labels.data) # 统计预测正确的样本数
        loss.append(sum_loss/len(ds))
        acc.append(float(train_correct)/len(ds))

    return loss, acc

if __name__ == '__main__':
    train_loader_std = load_data('西瓜数据集3.0.csv', 1) # 读数据，批大小为1
    train_loader_accm = load_data('西瓜数据集3.0.csv') # 读数据，批大小为所有样本

    epochs = 200 # 迭代次数为200，可以改
    accm_loss, accm_acc = learning(train_loader_accm)
    std_loss, std_acc = learning(train_loader_std)

    # 以下是绘制对比损失曲线
    fig = plt.figure(figsize=(6,4))
    plt.plot(range(epochs), std_loss, 'b-', label='standard BP')
    plt.plot(range(epochs), accm_loss, 'r-', label='accumulated BP')
    plt.legend()
    plt.title('std. BP VS accum. BP (loss)')
    plt.show()

'''
1、本题要求创建含1层隐层的多层感知机网络，分别按照标准BP（批大小为1，即每个样本调整一次权值），
   和按累计BP（批大小为全体样本，即计算所有样本的损失并累计，再集中一次调整权值）
2、多次运行本代码，统计观察标准BP和累计BP的损失曲线的变化
3、因为实例化的多层感知机Model权值初始化是随机的，所以每次运行loss曲线均有变化
4、即使标准BP和累计BP多层感知机网络的权值初始化也是随机，两者的随机权值也不同，所以两者下降速率表现也不同，
   可以考虑两者的权值初始化值取相同值，这样对比效果更好
5、隐层节点数是超参，取不同值情况下观察运行效果
6、因为本题仅仅是观察训练样本集上loss曲线的变化，故没有划分测试集
7、本题数据集规模太小（仅17个样本），所以准确率可能不尽理想   
'''