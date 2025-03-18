import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import os

from load_dataset import train_dataset, val_dataset

# 打印数据集信息
print(f"训练数据集长度:{len(train_dataset)}")
print(f"测试数据集长度:{len(val_dataset)}")

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 4)

# 初始化模型
model = model.to(device)

# 定义损失函数、优化器和学习率调度器
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

num_epochs = 100
best_accuracy = 0.0
best_model_state_dict = None
best_model_number = 0
accuracy = 0

# 创建 TensorBoard 日志
writer = SummaryWriter('/root/autodl-tmp/project/logs_train/logs_ResNet18')

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    training_loss = 0.0
    for train_data in train_dataloader:
        image, label = train_data
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {training_loss / len(train_dataloader)}")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_data in val_dataloader:
            image, label = val_data
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = loss_function(output, label)
            val_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_dataloader)}, Accuracy: {accuracy}%")

    # 调整学习率
    scheduler.step(val_loss / len(val_dataloader))

    # 记录训练和验证数据到 TensorBoard
    writer.add_scalar('Train Loss', training_loss / len(train_dataloader), epoch + 1)
    writer.add_scalar('Val Loss', val_loss / len(val_dataloader), epoch + 1)
    writer.add_scalar('Val Accuracy', accuracy, epoch + 1)

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state_dict = model.state_dict()
        best_model_number = epoch + 1

# 保存最佳模型权重
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, '/root/autodl-tmp/project/model_test/best_ResNet18.pth')
    print(f"模型已保存,该模型在epoch={best_model_number}时取得,在验证集上的准确率为{best_accuracy}")

torch.save(model.state_dict(), '/root/autodl-tmp/project/model_test/final_ResNet18.pth')
print(f"最终模型已保存, 在验证集上的准确率为{accuracy}%")

writer.close()

# 启动tensorboard命令：python -m tensorboard.main --logdir=logs_train/logs_ResNet18