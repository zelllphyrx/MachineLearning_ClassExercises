import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):  # 读取数据
    dataset = pd.read_csv(filename, delimiter=',', encoding='utf-8')
    # process the dataset
    attributeMap = {}  # 建立空的属性文字到数值的映射字典
    # 色泽: 3个属性值， 分别赋0, 0.5, 1数值
    attributeMap['浅白'] = 0
    attributeMap['青绿'] = 0.5
    attributeMap['乌黑'] = 1
    # 根蒂
    attributeMap['蜷缩'] = 0
    attributeMap['稍蜷'] = 0.5
    attributeMap['硬挺'] = 1
    # 敲声
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

    del dataset['编号']  # 删除无用的‘编号’
    dataset = np.array(dataset)
    m, n = np.shape(dataset)

    for i in range(m):
        for j in range(n):
            if dataset[i, j] in attributeMap:
                dataset[i, j] = attributeMap[dataset[i, j]]

    X = (dataset[:, :-1]).astype(np.float64)
    y = (dataset[:, -1]).astype(np.int8)

    return X, y

# 可视化
def plot_svc(model, ax=None, plot_support=True):
    # Plot the decision function for a 2D SVC
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)

    # create grid-points and coordinate matrix
    Y, X = np.meshgrid(y, x)
    # 堆叠数组
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='r', levels=[-1, 0, 1],
               alpha=0.5, linestyles=['--', '-', '--'])  # 生成支持面

    # plot supprt vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   alpha=0.3, s=300, linewidth=1, facecolors='y')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

if __name__ == '__main__':
    X, y = load_data('西瓜数据集3.0a.csv')
    linearSVC = SVC(kernel='linear')  # 线性核
    gaussSVC = SVC(kernel='rbf')  # 高斯核
    linearSVC.fit(X, y)
    gaussSVC.fit(X, y)
    print('linearSVC acc. = {:.2f}%'.format(100*linearSVC.score(X,y)))
    print('linearSVC support number: {}'.format(linearSVC.n_support_)) # 各类支持向量的个数   
    print('gaussSVC acc. = {:.2f}%'.format(100*gaussSVC.score(X,y)))
    print('gaussSVC support number: {}'.format(gaussSVC.n_support_)) # 各类支持向量的个数

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.xlim([-0, 1.2])
    plt.ylim([-0.3, 2])
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='jet')
    plot_svc(linearSVC)
    plt.title('Linear kernel SVC')
    plt.subplot(122)
    plt.xlim([-0, 1.2])
    plt.ylim([-0.3, 2])
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='jet')
    plot_svc(gaussSVC)
    plt.title('Gaussian kernel SVC')
    # plt.show()
    plt.savefig('result.png')