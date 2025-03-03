import os
import numpy as np
import librosa
import torch
from collections import Counter

# 参数设定
DATASET_DIR = "C:/Users/hyc49/Desktop/p3"  
OUTPUT_FILE = "audio_features.pth"  # 使用 .pth 扩展名便于 torch.save
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 40  
FIXED_TIME_STEPS = 64  

def extract_features(audio_path):
    """读取 WAV 文件并提取增强的音频特征"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # 提取 Mel 频谱图
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  

    # 提取 MFCC 及其一阶、二阶差分
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)  
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)  

    # 新增特征：色度图、频谱对比度、调式网络
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # 合并所有特征
    feature_matrix = np.vstack([mel_spec, mfcc, mfcc_delta, mfcc_delta2, chroma, spectral_contrast, tonnetz])  

    # 确保时间步长一致
    if feature_matrix.shape[1] > FIXED_TIME_STEPS:
        feature_matrix = feature_matrix[:, :FIXED_TIME_STEPS]  
    else:
        pad_width = FIXED_TIME_STEPS - feature_matrix.shape[1]
        feature_matrix = np.pad(feature_matrix, ((0, 0), (0, pad_width)), mode='constant')  

    # 标准化归一化
    feature_matrix = (feature_matrix - np.mean(feature_matrix)) / np.std(feature_matrix)

    return feature_matrix

def process_dataset(dataset_type):
    """处理 train/val 数据集"""
    audio_dir = os.path.join(DATASET_DIR, dataset_type)
    label_file = os.path.join(audio_dir, "labels.txt")

    # 读取标签
    with open(label_file, "r") as f:
        labels = [int(line.strip()) for line in f]

    # 获取所有 .wav 文件，确保排序一致
    filenames = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

    features = []
    num_files = len(filenames)
    processed_files = 0

    print(f"\n📌 处理 {dataset_type} 数据集，共 {num_files} 个样本...")

    for i, file in enumerate(filenames):
        audio_path = os.path.join(audio_dir, file)
        if os.path.exists(audio_path):
            feature_matrix = extract_features(audio_path)
            features.append({
                "filename": file,
                "feature_matrix": feature_matrix,
                "label": labels[i]
            })
            processed_files += 1
        else:
            print(f"⚠️ 警告: 找不到 {audio_path}，跳过该文件。")

        if (i + 1) % 10 == 0 or (i + 1) == num_files:
            print(f"  ✅ 进度: {i + 1}/{num_files} ({(i + 1) / num_files * 100:.1f}%)")

    print(f"🎯 {dataset_type} 数据集处理完成，共处理 {processed_files} 个样本。\n")
    return features, labels

if __name__ == "__main__":
    # 处理 train 和 val 数据集
    train_features, train_labels = process_dataset("train")
    val_features, val_labels = process_dataset("val")

    # 统计类别分布
    label_counts = Counter(train_labels + val_labels)
    print("📊 类别分布：", label_counts)

    # 转换为 PyTorch Tensor 格式
    train_features = [{"feature_matrix": torch.tensor(item["feature_matrix"], dtype=torch.float32), "label": item["label"]}
                      for item in train_features]
    val_features = [{"feature_matrix": torch.tensor(item["feature_matrix"], dtype=torch.float32), "label": item["label"]}
                    for item in val_features]

    # 保存数据
    torch.save({"train": train_features, "val": val_features}, OUTPUT_FILE)
    print(f"✅ 特征提取完成，已保存至 {OUTPUT_FILE}\n")
