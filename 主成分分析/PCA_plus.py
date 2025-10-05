from sklearn.decomposition import PCA
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

def load_faces(path):
    face_data = []
    face_files = os.listdir(path)
    face_files.sort()
    
    labels = [int(fn.split('.')[0][-2:])-1 for fn in face_files] # 为每个样本产生标签

    for fname in face_files:
        sub_path = os.path.join(path, fname)
        img = io.imread(sub_path)
        face_data.append(img.reshape(1, -1)[0]) # 将243*320展平为一维，即77760

    return np.array(face_data), np.array(labels)

def plot_faces(face_data, title, indices=None, shape=(243, 320), cmap='gray'):
    plt.figure(tight_layout=True)
    n_plot = 1
    n_rows, n_cols = 3, 4
    if indices is None:
        indices = range(face_data.shape[0])

    for i in indices:
        plt.subplot(n_rows, n_cols, n_plot)
        plt.imshow(face_data[i].reshape(shape), cmap=cmap)
        plt.axis('off')
        n_plot += 1
    plt.suptitle(title)
    plt.show()

if __name__ == '__main__':
    face_path = 'D:\Machine Learning\yalefaces'
    face_data, labels = load_faces(face_path)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(face_data, labels, test_size=0.2, random_state=42)

    dimensions = [5, 10, 20, 50, 100] # 不同的PCA维度
    for dim in dimensions:
        pca = PCA(n_components=dim)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # 使用KNN进行分类
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_pca, y_train)
        y_pred_knn = knn.predict(X_test_pca)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        print(f"KNN with PCA dim={dim}, Accuracy: {acc_knn:.4f}")
        print(classification_report(y_test, y_pred_knn))

        # 使用SVM进行分类
        svm = SVC(kernel='linear')
        svm.fit(X_train_pca, y_train)
        y_pred_svm = svm.predict(X_test_pca)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        print(f"SVM with PCA dim={dim}, Accuracy: {acc_svm:.4f}")
        print(classification_report(y_test, y_pred_svm))

    # 可视化每个人的第一张图像和均值图像
    face_index = np.arange(0, 165, 11)
    plot_faces(face_data[face_index], 'Original first face of the 15 people')
    
    pca = PCA(n_components=20)
    pca_faces = pca.fit_transform(face_data)
    reconstructed_faces = pca.inverse_transform(pca_faces)

    plot_faces(reconstructed_faces[face_index], 'Recomposed first face of the 15 people')

    mean_face = pca.mean_.reshape(243, 320)
    plt.figure(tight_layout=True)
    plt.imshow(mean_face, cmap='gray')
    plt.title('Mean face')
    plt.axis('off')
    plt.show()
