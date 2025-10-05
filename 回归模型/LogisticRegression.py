import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

def load_data(filename):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        c2n={'是':1, '否':0}
        density = [] # 密度
        sugar = [] # 含糖率
        y = [] # 标签
        data = csv.DictReader(csvfile)
        for item in data:
            density.append(float(item['密度']))
            sugar.append(float(item['含糖率']))
            y.append(c2n[item['好瓜']])

        density = np.array(density)
        sugar = np.array(sugar)
        X = np.hstack((density.reshape(-1,1), sugar.reshape(-1,1)))
    return (np.array(X), np.array(y))

if __name__ == '__main__':
    X, y = load_data('watermelon data set3.0a.csv') # 读入数据并生成X和y
    lr = LogisticRegression(penalty='none', max_iter=500, solver='sag') # 逻辑回归实例化，用了随机梯度下降法
    lr.fit(X, y) # 训练（拟合）
    y_pred = lr.predict(X) # 预测

    print(classification_report(y, y_pred, target_names=['是','否'])) # 显示测试结果

    # 以下是绘图
    fig = plt.figure(figsize=(6,4), tight_layout=True)
    plot_decision_regions(X, y, clf=lr)
    plt.xlim([min(X[:,0])-0.05, max(X[:,0])+0.05])
    plt.ylim([min(X[:,1])-0.05, max(X[:,1])+0.05])
    plt.xticks=([])
    plt.yticks=([])
    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.show()

    # 以下是制表及wb值
    print('w:', lr.coef_)
    print('b:', lr.intercept_)