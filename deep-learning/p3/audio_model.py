import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_FILE = "audio_features.pth"

# ============== 数据增强相关选项 ==============
AUDIO_TIME_SHIFT = True   # 是否启用随机时间偏移
MAX_SHIFT = 1600          # ~0.1秒(以16kHz为采样率)
NOISE_FACTOR = 0.005      # 随机噪声系数

# SpecAugment 参数
SPEC_TIME_MASKS = 1
SPEC_FREQ_MASKS = 1
TIME_MASK_PARAM = 15   # 遮挡多少帧
FREQ_MASK_PARAM = 8    # 遮挡多少个频带

def add_noise(waveform, noise_factor=NOISE_FACTOR):
    noise = torch.randn_like(waveform)
    return waveform + noise_factor * noise

def time_shift(waveform, max_shift=MAX_SHIFT):
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        waveform = torch.cat([waveform[shift:], torch.zeros(shift, device=waveform.device)], dim=0)
    elif shift < 0:
        shift = abs(shift)
        waveform = torch.cat([torch.zeros(shift, device=waveform.device), waveform[:-shift]], dim=0)
    return waveform

def spec_augment(mfcc, time_mask_param=TIME_MASK_PARAM, freq_mask_param=FREQ_MASK_PARAM,
                 num_time_masks=SPEC_TIME_MASKS, num_freq_masks=SPEC_FREQ_MASKS):
    """
    对 [batch, n_mfcc, time] 的 MFCC 做简单的时间遮挡 / 频率遮挡
    """
    # mfcc: [n_mfcc, time]
    n_mfcc, time_steps = mfcc.shape
    # 频率遮挡
    for _ in range(num_freq_masks):
        f0 = random.randint(0, n_mfcc - freq_mask_param)
        mfcc[f0:f0 + freq_mask_param, :] = 0
    # 时间遮挡
    for _ in range(num_time_masks):
        t0 = random.randint(0, time_steps - time_mask_param)
        mfcc[:, t0:t0 + time_mask_param] = 0
    return mfcc

class AudioDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data_list = data_list
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        waveform = item["waveform"]  # 1D, shape [num_samples]
        label = item["label"] - 1    # 转为0-based

        if self.augment:
            # 50% 概率加噪
            if random.random() < 0.5:
                waveform = add_noise(waveform)
            # 50% 概率时间偏移
            if AUDIO_TIME_SHIFT and random.random() < 0.5:
                waveform = time_shift(waveform)

        return waveform, label

# ============== 自注意力池化(Wav2Vec2输出) ==============
class SelfAttentionPool(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPool, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [batch, time, dim]
        attn_weights = torch.softmax(self.attn(x), dim=1)  # [batch, time, 1]
        pooled = torch.sum(x * attn_weights, dim=1)        # [batch, dim]
        return pooled

# ============== 多分辨率MFCC(40 & 80) + 1D卷积 + SpecAugment ==============
class MFCCBranch(nn.Module):
    """
    每个分辨率MFCC都用类似的结构:
      - MFCC提取
      - (可选) SpecAugment
      - 全局均值分支 + 卷积分支
      - 残差加和得到 128维
    """
    def __init__(self, sample_rate, n_mfcc=40):
        super(MFCCBranch, self).__init__()
        self.n_mfcc = n_mfcc
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": n_mfcc,
                "center": False,
            }
        )
        # 全局均值分支
        self.fc_global = nn.Linear(n_mfcc, 128)
        # 卷积分支
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_mfcc, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # [batch, 128, 1]
        )

    def forward(self, waveform, apply_specaug=False):
        """
        waveform: [batch, num_samples]
        return: [batch, 128] -> 融合全局 & 卷积特征
        """
        mfcc = self.mfcc_transform(waveform)  # [batch, n_mfcc, time]

        # 如果需要SpecAugment，可以在 batch 维度遍历，对每条MFCC做随机遮挡
        if apply_specaug and self.training:
            # mfcc[i]: shape [n_mfcc, time]
            for i in range(mfcc.size(0)):
                mfcc[i] = spec_augment(mfcc[i])

        # 全局均值
        mfcc_mean = mfcc.mean(dim=-1)               # [batch, n_mfcc]
        global_feat = torch.relu(self.fc_global(mfcc_mean))  # [batch, 128]

        # 卷积分支
        conv_feat = self.conv(mfcc)  # [batch, 128, 1]
        conv_feat = conv_feat.squeeze(-1)  # [batch, 128]

        # 残差加和
        feat_128 = global_feat + conv_feat
        return feat_128

# ============== 主模型：完全冻结 Wav2Vec2 + 多分辨率MFCC ==============
class AudioClassifierFusion(nn.Module):
    def __init__(self, num_classes=25, freeze_feature_extractor=True):
        super(AudioClassifierFusion, self).__init__()

        # ----------- Wav2Vec2 (完全冻结) -----------
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_feature_extractor and hasattr(self.wav2vec2, "feature_extractor"):
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False

        hidden_dim = self.wav2vec2.config.hidden_size  # 768
        self.attn_pool = SelfAttentionPool(hidden_dim)

        # ----------- 多分辨率 MFCC 分支 -----------
        self.mfcc_branch_40 = MFCCBranch(sample_rate=16000, n_mfcc=40)
        self.mfcc_branch_80 = MFCCBranch(sample_rate=16000, n_mfcc=80)

        # 两个分支各输出128维 → total 256维
        # 再和 Wav2Vec2(768) 融合 → 768 + 256 = 1024
        fusion_dim = hidden_dim + 256
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(fusion_dim, num_classes)

    def forward(self, waveform):
        # Wav2Vec2 分支
        with torch.no_grad():
            outputs = self.wav2vec2(waveform)
        wav2vec_feats = self.attn_pool(outputs.last_hidden_state)  # [batch, 768]

        # MFCC 分支 (多分辨率)
        #   训练时对其中一个或两个都做 SpecAugment(可自行调节)
        mfcc_40 = self.mfcc_branch_40(waveform, apply_specaug=True)
        mfcc_80 = self.mfcc_branch_80(waveform, apply_specaug=True)

        # 拼接
        mfcc_merged = torch.cat([mfcc_40, mfcc_80], dim=1)  # [batch, 256]

        # 最终融合
        fused = torch.cat([wav2vec_feats, mfcc_merged], dim=1)  # [batch, 768+256=1024]
        fused = self.dropout(fused)
        logits = self.fc(fused)  # [batch, num_classes]
        return logits

def main():
    # 1. 加载音频特征数据
    data = torch.load(FEATURES_FILE)
    train_data = data["train"]
    val_data = data["val"]

    train_dataset = AudioDataset(train_data, augment=True)
    val_dataset = AudioDataset(val_data, augment=False)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    num_classes = 25
    model = AudioClassifierFusion(num_classes=num_classes, freeze_feature_extractor=True).to(DEVICE)

    # 3. 优化器 & 调度器 (与原代码保持一致)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    num_epochs = 50
    best_val_f1 = 0.0

    # 4. 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for waveforms, labels in train_loader:
            waveforms = waveforms.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * waveforms.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        # ===== 验证 =====
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms = waveforms.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * waveforms.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        # 调整LR
        scheduler.step(val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "audio_model_best.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
              f"Val F1: {val_f1:.4f}")

    torch.save(model.state_dict(), "audio_model_final.pth")
    print("✅ Training complete. Best Val F1:", best_val_f1)

if __name__ == "__main__":
    main()

