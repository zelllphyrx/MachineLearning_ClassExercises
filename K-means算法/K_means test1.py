from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
import numpy as np

'''在sklearn中，使用模块metrics中的类silhouette_score来计算轮廓系数，它返回的是一个数据集中所有样本的
轮廓系数的均值。但还有同在metrics模块中的silhouette_sample，它的参数与轮廓系数一致，但返回的是数据集中
每个样本的轮廓系数'''

X,y = make_blobs(n_samples=500, n_features=2, centers=4,
                 cluster_std=0.7, random_state=1)

clusters = [2,3,4,5,6,7] # 测试的聚类簇数，共6个
si_avgs = []

for i, n_clustes in enumerate(clusters):
    kmeans = KMeans(n_clusters=n_clustes, n_init=10)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    si_avgs.append(silhouette_score(X, cluster_labels))
    print('n_clusters = {}, si = {}'.format(n_clustes, si_avgs[i]))

best_n_clutsers = clusters[np.argmax(si_avgs)]
# print(best_n_clutsers)

kmeans = KMeans(n_clusters=best_n_clutsers, n_init=10)
kmeans.fit(X)
cluster_labels = kmeans.labels_

plt.figure(figsize=(10,6), tight_layout=True)
plt.subplot(1,2,1)
plt.bar(clusters, si_avgs)
plt.xlabel('n_cluster')
plt.ylabel('silhoutte score')
plt.title('silhoutte analysis, best_n_clusters = {}'.format(best_n_clutsers))

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c = cluster_labels, marker='o', alpha=0.3)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=150, c='r', marker='^', alpha=0.6)
plt.title('KMeans (K = {})'.format(best_n_clutsers))
plt.suptitle('KMeans: in fact, the best n_clusters should be %d' % best_n_clutsers, fontsize=20)
plt.savefig('result.png')