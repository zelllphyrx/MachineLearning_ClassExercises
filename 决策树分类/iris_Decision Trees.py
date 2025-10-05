from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import tree

iris = load_iris() # 读入iris数据集
x_train,x_test,y_train,y_test = train_test_split(iris.data, # 数据集划分训练集和测试集
                                                 iris.target,
                                                 test_size=0.5) # 测试集占比整个数据集为30%

clf = DecisionTreeClassifier(criterion='entropy', # 实例化分类决策树，'entropy'-信息增益， 'gini'-Gini指数
                             max_depth=3) # max_depth指定树的的深度
clf.fit(x_train,y_train) # 用训练集进行学习，是一个拟合
y_pred = clf.predict(x_test) # 用测试集预测，预测结果记录在y_pred中

print(classification_report(y_test,y_pred)) # 统计测试集上的测试结果

# 以下是显示决策树
fig = plt.figure(figsize=(10,6), tight_layout=True)
tree.plot_tree(clf,
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True)
plt.show()

'''
建议：
1、调整不同的测试集划分比例，观察和统计测试结果
2、criterion取'entropy‘和'gini’，对比统计测试结果
3、设置不同的树深度（max_depth），对比统计测试结果
4、选用除了iris数据集外的其他数据集，如wine，……
'''