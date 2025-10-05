from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits  # 导入手写数字数据集
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import tree

digits = load_digits()  # 加载手写数字数据集
x_train, x_test, y_train, y_test = train_test_split(digits.data,  # 将手写数字数据集划分为训练集和测试集
                                                    digits.target,
                                                    test_size=0.38)

clf = DecisionTreeClassifier(criterion='entropy',  # 使用 'entropy' 准则实例化 DecisionTreeClassifier
                             max_depth=8)
clf.fit(x_train, y_train)  # 在训练数据上拟合模型
y_pred = clf.predict(x_test)  # 对测试数据进行预测

print(classification_report(y_test, y_pred))  # 打印分类报告

# 绘制决策树
fig = plt.figure(figsize=(10, 6), tight_layout=True)
tree.plot_tree(clf,
               filled=True)
plt.show()
