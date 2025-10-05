import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
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
    X, y = load_data('西瓜数据集3.0a.csv') # 读入数据并生成X和y
    clf = lda()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, target_names=['是', '否']))  # 显示测试结果

    # 以下是绘图
    fig = plt.figure(figsize=(5, 5), tight_layout=True)
    plot_decision_regions(X, y, clf=clf)
    plt.xlim([min(X[:,0])-0.05, max(X[:,0])+0.05])
    plt.ylim([min(X[:,1])-0.05, max(X[:,1])+0.05])
    plt.xticks = ([])
    plt.yticks = ([])
    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.show()