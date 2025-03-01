import os
import torch
from feature_extraction import extract_features  # 从 feature_extraction.py 导入特征提取函数
from audio_model import AudioClassifierCNNTransformer  # 从 audio_model.py 导入模型类

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径设置
MODEL_PATH = "audio_model_best.pth"  # 训练好的模型权重文件
TEST_DIR = "test"                  # 存放测试音频文件的文件夹（应包含75个.wav文件）
OUTPUT_FILE = "prediction.txt"     # 输出预测结果的文本文件

def load_model():
    # 根据训练时的设置，特征维度为273，类别数为25
    feature_dim = 273
    num_classes = 25
    # 初始化模型（参数必须与训练时保持一致）
    model = AudioClassifierCNNTransformer(feature_dim, num_classes, d_model=128,
                                            num_conv_channels=64,
                                            num_transformer_layers=2,
                                            nhead=4, dropout=0.5)
    # 加载训练好的权重
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_audio(model, audio_path):
    """
    对单个音频文件进行特征提取和预测
    1. 调用 extract_features 提取特征，得到形状为 [273, 64] 的 numpy 数组
    2. 转换为 Tensor，并增加 batch 维度，得到形状 [1, 273, 64]
    3. 模型预测，返回预测类别（原标签从1开始）
    """
    # 提取特征
    feature = extract_features(audio_path)  # 形状：[273, 64]
    feature = torch.tensor(feature, dtype=torch.float32)
    feature = feature.unsqueeze(0).to(DEVICE)  # 增加 batch 维度
    with torch.no_grad():
        outputs = model(feature)  # 输出形状：[1, num_classes]
        _, pred = torch.max(outputs, 1)
    # 模型输出的索引从 0 开始，转换为 [1,25]
    return pred.item() + 1

def main():
    # 加载模型
    model = load_model()
    
    # 获取测试文件列表（按照文件名排序）
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".wav")])
    
    predictions = []
    print("开始预测测试集文件：")
    for file in test_files:
        audio_path = os.path.join(TEST_DIR, file)
        pred_label = predict_audio(model, audio_path)
        predictions.append(pred_label)
        print(f"文件 {file} 预测标签: {pred_label}")
    
    # 将预测结果保存到 prediction.txt，每行一个预测标签
    with open(OUTPUT_FILE, "w") as f:
        for label in predictions:
            f.write(f"{label}\n")
    
    print(f"预测完成，结果已保存在 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
