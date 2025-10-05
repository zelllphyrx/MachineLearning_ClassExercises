from sklearn.decomposition import PCA
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

def load_faces(path):
    face_data = []
    face_files = os.listdir(path)
    face_files.sort()
    
    labels = [int(fn.split('.')[0][-2:])-1 for fn in face_files]  # 为每个样本产生标签

    for fname in face_files:
        sub_path = os.path.join(path, fname)
        img = io.imread(sub_path)
        face_data.append(img.reshape(1, -1)[0])  # 将243*320展平为一维，即77760

    return face_data, labels

if __name__ == '__main__':
    face_path = 'D:\\Machine Learning\\yalefaces'
    face_data, labels = load_faces(face_path)

    pca = PCA(n_components=3)  # 前3维
    pca_faces = pca.fit_transform(face_data)  # 降维至3维后新的人脸，243*320->3  

    from mpl_toolkits.mplot3d import Axes3D

    plt.figure(figsize=(10, 8), tight_layout=True)
    ax = plt.subplot(111, projection='3d')
    scatter = ax.scatter(pca_faces[:, 0], pca_faces[:, 1], pca_faces[:, 2], c=labels, cmap='viridis', marker='o')

    # 手动构建图例
    handles, _ = scatter.legend_elements(num=15)
    labels_unique = [f'Person {i+1}' for i in range(15)]
    ax.legend(handles, labels_unique, title="People")

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.title('PCA 3D Visualization of Yale Faces')

    plt.show()
