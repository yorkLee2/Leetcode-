import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置设备和特征数据文件路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_FILE = "audio_features.pth"  # 该文件应包含 "train" 和 "val" 键

# SpecAugment：对输入特征（如Mel谱图）随机进行时间和频率掩码（数据增强）
def spec_augment(feature, time_mask_param=8, freq_mask_param=8, num_time_masks=2, num_freq_masks=2):
    # feature: numpy array或Tensor，形状 [feature_dim, time_steps]
    if isinstance(feature, torch.Tensor):
        feature = feature.clone().detach().cpu().numpy()
    augmented = feature.copy()
    num_mel_channels, num_frames = augmented.shape

    # 时间掩码
    for _ in range(num_time_masks):
        t = random.randrange(0, time_mask_param)
        t0 = random.randrange(0, max(1, num_frames - t))
        augmented[:, t0:t0+t] = 0

    # 频率掩码
    for _ in range(num_freq_masks):
        f = random.randrange(0, freq_mask_param)
        f0 = random.randrange(0, max(1, num_mel_channels - f))
        augmented[f0:f0+f, :] = 0

    return torch.tensor(augmented, dtype=torch.float32)

# 自定义数据集：支持数据增强（仅在训练时开启）
class AudioDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data_list = data_list
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        feature = item["feature_matrix"]
        label = item["label"] - 1  # 标签从 0 开始
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float32)
        # 若开启数据增强，则对输入特征（如Mel谱图）进行 SpecAugment
        if self.augment:
            feature = spec_augment(feature)
        return feature, label

# 定义基于 CNN + Transformer Encoder 的模型
class AudioClassifierCNNTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes, d_model=128, num_conv_channels=64,
                 num_transformer_layers=2, nhead=4, dropout=0.5):
        super(AudioClassifierCNNTransformer, self).__init__()
        # CNN部分：两层卷积，输出维度为 d_model
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=num_conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_conv_channels)
        self.conv2 = nn.Conv1d(in_channels=num_conv_channels, out_channels=d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Transformer Encoder部分：输入为 [time, batch, d_model]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, feature_dim, time_steps]
        x = self.conv1(x)      # -> [batch, num_conv_channels, time_steps]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)       # -> [batch, num_conv_channels, time_steps/2]

        x = self.conv2(x)      # -> [batch, d_model, time_steps/2]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)       # -> [batch, d_model, time_steps/4]

        # Transformer Encoder要求输入形状 [time, batch, d_model]
        x = x.transpose(1, 2)  # -> [batch, time_steps/4, d_model]
        x = x.transpose(0, 1)  # -> [time_steps/4, batch, d_model]
        x = self.transformer_encoder(x)  # -> [time_steps/4, batch, d_model]
        # 聚合：取时间步均值
        x = x.mean(dim=0)      # -> [batch, d_model]
        x = self.dropout(x)
        logits = self.fc(x)    # -> [batch, num_classes]
        return logits

def main():
    # 加载特征数据（必须包含 "train" 和 "val"）
    data = torch.load(FEATURES_FILE)
    train_data = data["train"]
    val_data = data["val"]

    # 开启数据增强仅在训练集使用
    train_dataset = AudioDataset(train_data, augment=True)
    val_dataset = AudioDataset(val_data, augment=False)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 获取输入特征维度（feature_dim），即第一维大小
    sample_feature, _ = train_dataset[0]
    feature_dim = sample_feature.shape[0]

    num_classes = 25
    # 这里d_model为Transformer的输入维度，建议与特征维度适当匹配
    d_model = 128
    num_transformer_layers = 2
    nhead = 4
    dropout = 0.5

    model = AudioClassifierCNNTransformer(feature_dim, num_classes, d_model=d_model,
                                            num_conv_channels=64,
                                            num_transformer_layers=num_transformer_layers,
                                            nhead=nhead, dropout=dropout)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    num_epochs = 200
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        for features, labels in train_loader:
            features = features.to(DEVICE)  # [batch, feature_dim, time_steps]
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        train_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                labels = labels.to(DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        # 根据验证 F1 更新学习率
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "audio_model_best.pth")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train F1-Score: {train_f1:.4f}, Validation F1-Score: {val_f1:.4f}")

    torch.save(model.state_dict(), "audio_model_final.pth")
    print("✅ 模型训练完成，已保存最佳模型至 audio_model_best.pth 和最终模型至 audio_model_final.pth")

if __name__ == "__main__":
    main()
