'''
Yale数据集，共有15个人的人脸图像，每个人11张图像，图像大小320*243=77760
'''
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
    
    labels=[int(fn.split('.')[0][-2:])-1 for fn in face_files] #为每个样本产生标签

    for fname in face_files:
        sub_path = os.path.join(path, fname)
        img = io.imread(sub_path)
        face_data.append(img.reshape(1,-1)[0]) # 将243*320展平为一维，即77760

    return face_data, labels

if __name__ == '__main__':
    face_path = 'D:\Machine Learning\yalefaces'
    face_data, labels = load_faces(face_path)

    pca = PCA(n_components=3) #前20维
    pca_faces = pca.fit_transform(face_data) #降维至20维后新的人脸，243*320->20  

    plt.figure(1,tight_layout=True)
    n_plot=1
    for i in range(11): #第1个人的11张照片
        plt.subplot(3,4,n_plot)
        plt.imshow(face_data[i].reshape(243,320),cmap='gray')
        plt.axis('off')
        n_plot+=1
    plt.suptitle('11 photos of first person')

    #筛选每个人的第一张图像做可视化
    #人脸编号
    face_index = np.arange(0,165,11)

    plt.figure(2,tight_layout=True)

    n_plot = 1
    for i in face_index:
        plt.subplot(4,5,n_plot)
        plt.imshow(face_data[i].reshape(243,320),cmap='gray')
        plt.title('{}'.format(n_plot))
        plt.axis('off')

        n_plot+=1
    plt.suptitle('original first face of the 15 people')    

    plt.figure(3,tight_layout=True)
    n_plot = 1
    for i in face_index:
        plt.subplot(3,5,n_plot)
        re_face = pca.inverse_transform(pca_faces[i]).reshape(243,320) #重构, 20->243*320
        plt.imshow(re_face,cmap='gray')
        plt.title('{}'.format(n_plot))
        plt.axis('off')

        n_plot+=1

    plt.suptitle('recomposition first face of the 15 people')
    # plt.savefig('result.png')

    plt.figure(4,tight_layout=True)
    mean_face = pca.mean_.reshape(243,320)
    plt.imshow(mean_face,cmap='gray')
    plt.title('mean face')
    plt.axis('off')

    plt.show()