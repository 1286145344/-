import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


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

# 标签one-hot编码
lb = LabelBinarizer()
train_labels_one_hot = lb.fit_transform(train_labels)
test_labels_one_hot = lb.transform(test_labels)

# 降维 (PCA) 用于加速训练, 设定维度为50以减少计算量
pca = PCA(n_components=50)  # 可以进一步尝试调整维度以提高速度
train_data_pca = pca.fit_transform(train_data.reshape(-1, 32 * 32 * 3))
test_data_pca = pca.transform(test_data.reshape(-1, 32 * 32 * 3))

# 训练SVM (使用 LinearSVC 替代 SVC)
svm = LinearSVC(C=1, max_iter=1000, random_state=42, multi_class='ovr', tol=1e-3, verbose=0, dual=False)


# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


labels = [str(i) for i in range(10)]

# 训练和评估 SVM
svm.fit(train_data_pca, train_labels)

# 预测
svm_preds = svm.predict(test_data_pca)

# 输出分类报告
print("SVM Classification Report:")
print(classification_report(test_labels, svm_preds))

# 可视化混淆矩阵
cm = confusion_matrix(test_labels, svm_preds)
plot_confusion_matrix(cm, labels)


# 训练过程中记录训练和验证准确率（通过分批训练）
def plot_accuracy_curve(model, train_data, train_labels, test_data, test_labels, model_name, batch_size=200):
    train_accuracy_list = []
    test_accuracy_list = []

    # 分批次训练
    n_samples = train_data.shape[0]
    for i in range(0, n_samples, batch_size):
        X_batch = train_data[i:i + batch_size]
        y_batch = train_labels[i:i + batch_size]

        # 每次训练一个batch
        model.fit(X_batch, y_batch)

        # 计算训练集和验证集的准确率
        train_preds = model.predict(X_batch)
        test_preds = model.predict(test_data)

        train_accuracy = accuracy_score(y_batch, train_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)

        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracy_list, label='Training Accuracy')
    plt.plot(test_accuracy_list, label='Test Accuracy')
    plt.title(f'{model_name} Accuracy Curve')
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# 可视化SVM训练准确率
plot_accuracy_curve(svm, train_data_pca, train_labels, test_data_pca, test_labels, 'SVM')
