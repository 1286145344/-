import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=0)  # 使用并行化
rf.fit(train_data.reshape(-1, 32 * 32 * 3), train_labels)

# 预测和评估 随机森林
rf_preds = rf.predict(test_data.reshape(-1, 32 * 32 * 3))
print("Random Forest Classification Report:")
print(classification_report(test_labels, rf_preds))

# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

labels = [str(i) for i in range(10)]
cm = confusion_matrix(test_labels, rf_preds)
plot_confusion_matrix(cm, labels)

# 可视化随机森林训练准确率曲线
def plot_accuracy_curve(model, train_data, train_labels, test_data, test_labels, model_name):
    # 模拟训练过程中的训练和测试准确率变化（由于RandomForest是非增量学习模型，所以无法直接画出准确率变化）
    # 这里会比较训练和测试集的最终准确率
    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)

    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)

    print(f'{model_name} Training Accuracy: {train_accuracy:.4f}')
    print(f'{model_name} Test Accuracy: {test_accuracy:.4f}')

    # 模拟训练过程中的准确率变化（由于RandomForest是非增量学习模型，无法直接绘制）
    # 直接展示训练前后的准确率
    return train_accuracy, test_accuracy

# 随机森林训练曲线
train_accuracy, test_accuracy = plot_accuracy_curve(
    rf,
    train_data.reshape(-1, 32 * 32 * 3),
    train_labels,
    test_data.reshape(-1, 32 * 32 * 3),
    test_labels,
    'Random Forest'
)

# 可视化训练准确率和测试准确率曲线
def plot_train_test_accuracy_curve(train_acc, test_acc):
    # 使用matplotlib绘制准确率曲线
    epochs = [0, 1]  # 这里只能绘制静态值，增加更多轮次需要自定义训练过程
    plt.plot(epochs, [train_acc, train_acc], label='Train Accuracy')
    plt.plot(epochs, [test_acc, test_acc], label='Test Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Test Accuracy Curve')
    plt.show()

plot_train_test_accuracy_curve(train_accuracy, test_accuracy)
