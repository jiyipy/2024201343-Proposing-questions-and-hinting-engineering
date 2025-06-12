import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# 参数设置
# 随机种子
RANDOM_SEED = 42
# BATCH_SIZE指在一次模型训练迭代中使用的训练样本数量
BATCH_SIZE = 16
# 合适的学习率能帮助模型更快地找到全局最小值或接近最优解，避免过拟合
LEARNING_RATE = 0.01
# 是深度学习中用于控制模型训练周期的超参数
NUM_EPOCHS = 100
#设置隐层神经单元数量
HIDDEN_SIZE = 10

# 设置随机种子（保证可重复性）
torch.manual_seed(RANDOM_SEED)

# 1. 数据准备
# --------------------------------------------------
# 加载数据集

df=pd.read_csv('../data/manufacturing_defect_dataset.csv')

X=df.iloc[:,:-1]
# y = df.iloc[:, -1]
# le = LabelEncoder()
# y = le.fit_transform(y)
y=df.iloc[:,-1].values

# 数据预处理，特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# 转换为Tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# 创建DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 2. 定义神经网络模型
# --------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.layer1_1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()


        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)

        out = self.layer1_1(out)
        out = self.relu(out)


        out = self.layer2(out)
        return out

# 初始化模型
input_size = X_train.shape[1]
num_classes = len(torch.unique(y_train))
model = NeuralNet(input_size, HIDDEN_SIZE, num_classes)

# 3. 定义损失函数和优化器
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. 训练模型
# --------------------------------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)

    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
              f'Loss: {total_loss / len(train_loader):.4f}, '
              f'Test Accuracy: {accuracy:.4f}')

# 5. 最终测试
# --------------------------------------------------
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
print(f'\nFinal Test Accuracy: {accuracy:.4f}')

print(classification_report(predicted,y_test))

# 6. 保存模型（可选）
# torch.save(model.state_dict(), 'iris_classifier.pth')