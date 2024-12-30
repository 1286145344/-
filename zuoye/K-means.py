import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, classification_report, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 加载CIFAR-10数据集
def load_cifar10_batch(batch_filename):
    cifar10_folder = 'cifar-10'
    full_path = os.path.join(cifar10_folder, batch_filename)
    with open(full_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    return data, labels

# 加载所有数据
def load_cifar10_data():
    data, labels = [], []
    for i in range(1, 6):
        batch_filename = f"data_batch_{i}"
        batch_data, batch_labels = load_cifar10_batch(batch_filename)
        data.append(batch_data)
        labels.append(batch_labels)
    test_batch_filename = "test_batch"
    test_data, test_labels = load_cifar10_batch(test_batch_filename)

    # 将数据合并成一个大数组
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels, test_data, test_labels

# 加载数据集
train_data, train_labels, test_data, test_labels = load_cifar10_data()

# 数据标准化
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# 降维 (PCA) 用于加速计算，设定维度为50以减少计算量
pca = PCA(n_components=50)
train_data_pca = pca.fit_transform(train_data.reshape(-1, 32 * 32 * 3))
test_data_pca = pca.transform(test_data.reshape(-1, 32 * 32 * 3))

# 初始化KMeans
kmeans = KMeans(n_clusters=10, random_state=0)

# 手动迭代并记录ARI
ari_scores = []
n_iter = 100  # 设置迭代次数
for i in range(n_iter):
    kmeans.fit(train_data_pca)
    ari_scores.append(adjusted_rand_score(train_labels, kmeans.labels_))

# 聚类预测
kmeans_preds = kmeans.predict(test_data_pca)

# 无监督评估
ari_score = adjusted_rand_score(test_labels, kmeans_preds)
print(f"Adjusted Rand Index for KMeans Clustering: {ari_score:.4f}")

# 计算分类准确率、召回率、精确率
accuracy = accuracy_score(test_labels, kmeans_preds)
precision = precision_score(test_labels, kmeans_preds, average='weighted')
recall = recall_score(test_labels, kmeans_preds, average='weighted')
report = classification_report(test_labels, kmeans_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print("Classification Report:\n", report)

# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# 可视化KMeans的聚类结果
labels = [str(i) for i in range(10)]
cm = confusion_matrix(test_labels, kmeans_preds)
plot_confusion_matrix(cm, labels)

# 可视化训练过程中ARI的变化
plt.figure(figsize=(8, 6))
plt.plot(range(len(ari_scores)), ari_scores, marker='o', color='b', label='ARI')
plt.title("ARI During Training")
plt.xlabel("Iteration")
plt.ylabel("ARI")
plt.legend()
plt.grid(True)
plt.show()
