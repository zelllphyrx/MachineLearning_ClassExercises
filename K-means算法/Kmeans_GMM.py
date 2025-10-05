from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename, header=None, delimiter=',', encoding='utf-8')    
    return data

if __name__ == '__main__':
    X = load_data('D:\Machine Learning\西瓜数据集4.0.csv')
    # print(X)
    K = 3 #簇个数
    gmm = GaussianMixture(n_components=K, random_state=10)
    gmm.fit(X.values[:,:2])
    y = gmm.predict(X.values[:,:2])
    cluster_centers = gmm.means_

    print(cluster_centers) # 簇中心坐标

    # visualization
    fig = plt.figure(figsize=(10,10))
    plt.scatter(X.values[:,0], X.values[:,1], c=y, s=150)    
    plt.xlabel('density',fontsize=15)
    plt.ylabel('sugar',fontsize=15)
    plt.title('$K={}$'.format(K),fontsize=20)
    for k in range(K):
        plt.scatter(cluster_centers[k][0], cluster_centers[k][1],
                    marker='X', c='r', s=150)

    plt.savefig('result_gmm.png')
    
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 计算轮廓系数
silhouette = silhouette_score(X.values[:,:2], y)

# 计算戴维斯-布尔丁指数
davies_bouldin = davies_bouldin_score(X.values[:,:2], y)

# 计算 Calinski-Harabasz 指数
calinski_harabasz = calinski_harabasz_score(X.values[:,:2], y)

print("Silhouette Score:", silhouette)
print("Davies-Bouldin Score:", davies_bouldin)
print("Calinski-Harabasz Score:", calinski_harabasz)
