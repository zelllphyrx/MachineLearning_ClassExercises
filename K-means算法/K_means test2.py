from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

X,y = make_blobs(n_samples=500, n_features=2, centers=4,
                 cluster_std=1.5, random_state=1)

clusters = [2,3,4,5,6,7] # 测试的聚类簇数，共6个
sses = []

for i, n_clustes in enumerate(clusters):
    kmeans = KMeans(n_clusters=n_clustes, n_init=10)
    kmeans.fit(X)
    sses.append(kmeans.inertia_) #获取sse值
    # print(kmeans.inertia_)

kneeLoc = KneeLocator(range(0,6), # 查找拐点
                              sses,
                              curve='convex',
                              direction='decreasing')

best_n_clutsers = kneeLoc.elbow+2 #因为sses的下标从0开始，且sees首元素是2，所以实际的best_n_clutsers需要+2
sse=kneeLoc.elbow_y

# print(best_n_clutsers)

kmeans = KMeans(n_clusters=best_n_clutsers, n_init=10)
kmeans.fit(X)
cluster_labels = kmeans.labels_

plt.figure(figsize=(10,6), tight_layout=True)
plt.subplot(1,2,1)
plt.plot(clusters,sses,'ro--',label='RSS')
plt.scatter(best_n_clutsers,sse,marker='o',c='g',s=200, alpha=0.5,label='knee point\nbest n_clusters')
plt.xlabel('n_cluster')
plt.ylabel('RSS')
plt.legend()
plt.title('RSS, best_n_clusters = {}'.format(best_n_clutsers))

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c = cluster_labels, marker='o', alpha=0.3)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=150, c='r', marker='^', alpha=0.6,label='centeroid')
plt.title('KMeans (K = {})'.format(best_n_clutsers))
plt.legend()
plt.suptitle('KMeans: in fact, the best n_clusters should be %d' % best_n_clutsers, fontsize=20)

plt.savefig('result.png')