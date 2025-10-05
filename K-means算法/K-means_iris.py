from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as si, davies_bouldin_score as db, calinski_harabasz_score as ch
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
n_classes = len(iris.target_names)

scores = []
num_k = [2, 3, 4, 5]
for k in num_k:
    kmeans = KMeans(n_clusters=k, random_state=1, n_init=50).fit(X)
    labels = kmeans.labels_
    scores.append([si(X, labels),
                   db(X, labels),
                   ch(X, labels)])

# 对 calinski_harabasz_score 做归一化
ch_score_norm = [score[2] for score in scores]
min_score = min(ch_score_norm)
max_score = max(ch_score_norm)
ch_score_norm = [(s - min_score) / (max_score - min_score) for s in ch_score_norm]

plt.figure(figsize=(7,4),tight_layout=True)
plt.plot(num_k, [score[0] for score in scores], 'ro--', alpha=0.7, label='silhouette_score(↑)')
plt.plot(num_k, [score[1] for score in scores], 'bo--', alpha=0.7, label='davies_bouldin_score(↓)')
plt.plot(num_k, ch_score_norm, 'go--', alpha=0.7, label='calinski_harabasz_score(↑)')
plt.xticks(num_k)
plt.legend()
plt.grid()
plt.title('Different k values with KMeans on iris dataset\nTrue classes=%d' % n_classes)

plt.savefig('result_kmeans_iris.png')
plt.show()