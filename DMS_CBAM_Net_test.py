#模型的测试和性能评估
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix

from load_dataset import test_dataset
from model.DMS_CBAM_Net import DMS_CBAM_Net

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DMS_CBAM_Net().to(device)
model.load_state_dict(torch.load("final_DMS_CBAM_Net.pth"))
model.eval()

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
loss_function = nn.CrossEntropyLoss()

test_loss = 0.0
correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for test_data in test_dataloader:
        image,label = test_data
        image = image.to(device)
        label = label.to(device)
        output = model(image)

        loss = loss_function(output,label)
        test_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        #统计所有预测的结果和真实标签
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    # 性能指标:准确率、精确率、召回率、F1分数、混淆矩阵
    accuracy = 100 * correct / total
    precision = 100 * precision_score(all_labels, all_predictions, average='weighted')
    recall = 100 *  recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)

    # precision = precision_score(all_labels, all_predictions, average='macro')
    # recall = recall_score(all_labels, all_predictions, average='macro')
    # f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"测试集上的损失值Test Loss:{test_loss}")
    print(f"准确率Accuracy:{accuracy}%")
    print(f"精确率Precision: {precision}%")
    print(f"召回率Recall: {recall}%")
    print(f"F1分数F1 Score: {f1}")

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Bacterialblight", "Blast", "Brownspot", "Tungro"],
                yticklabels=["Bacterialblight", "Blast", "Brownspot", "Tungro"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()