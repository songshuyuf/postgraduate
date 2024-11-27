#!/usr/bin/env python
# coding=utf-8
import os
import librosa
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from transformers import Wav2Vec2Processor
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm



# 数据处理
def load_and_extract_features(audio_path, processor):
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest")
        input_values = inputs["input_values"].squeeze().tolist()
        return input_values, sr
    except Exception as e:
        print(f"处理文件 {audio_path} 时出错: {e}")
        return None, None

def preprocess_directory(directory_path, processor):
    features = []
    labels = []

    for root, dirs, files in os.walk(directory_path):
        label_file_path = os.path.join(root, 'label.txt')
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r', encoding='utf-8') as f:
                data_label = f.read().strip()
            try:
                # 尝试将标签转换为数值类型
                data_label = float(data_label)  # 或 int(data_label) 根据实际需求
            except ValueError:
                print(f"标签 {data_label} 无法转换为数字。")
                continue  # 跳过该目录

        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                feature, _ = load_and_extract_features(file_path, processor)
                if feature is not None:
                    features.append(feature)
                    labels.append(data_label)
                else:
                    print(f"提取特征失败: {file_path}")

    return features, labels

def create_dataset(features, labels):
    if len(features) == 0 or len(labels) == 0:
        print("没有找到特征或标签。")
        return None

    print(f"标签样例: {labels[:1]}")
    try:
        dataset = Dataset.from_dict({"input_values": features, "label": labels})
        print("创建数据集成功！")  # 显示创建数据集成功的提示
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        return None
    return dataset

def pad_features(features, max_length):
    padded_features = []
    for feature in features:
        if len(feature) < max_length:
            feature = feature + [0] * (max_length - len(feature))
        else:
            feature = feature[:max_length]
        padded_features.append(feature)
    return torch.tensor(padded_features, dtype=torch.float32)

# 定义模型
class SentimentGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SentimentGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for features_batch, labels_batch in train_loader:
                features_batch = features_batch.float()
                features_batch = features_batch.unsqueeze(1)

                outputs = model(features_batch)
                loss = criterion(outputs.squeeze(), labels_batch.float())

                # 计算准确率
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                train_correct += (predicted.squeeze() == labels_batch.float()).sum().item()
                train_total += labels_batch.size(0)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item(), 'accuracy': train_correct / train_total})
                pbar.update(1)
        
        # 验证模式
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.float()
                features_batch = features_batch.unsqueeze(1)

                outputs = model(features_batch)
                loss = criterion(outputs.squeeze(), labels_batch.float())

                # 计算准确率
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (predicted.squeeze() == labels_batch.float()).sum().item()
                val_total += labels_batch.size(0)
                val_loss += loss.item()

        # 计算平均损失和准确率
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        # 打印训练和验证的损失与准确率
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


def load_model(model_path, input_size, hidden_size, num_layers):
    model = SentimentGRU(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()
    return model

def predict(audio_path, processor, model, scaler, max_length):
    feature, _ = load_and_extract_features(audio_path, processor)
    if feature is not None:
        # 标准化特征
        feature_np = np.array(feature).reshape(1, -1)
        feature_scaled = scaler.transform(feature_np)
        feature_scaled = feature_scaled.tolist()

        feature_padded = pad_features([feature_scaled[0]], max_length)
        feature_tensor = torch.tensor(feature_padded, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(feature_tensor)
            predicted_score = torch.sigmoid(output).item()
            predicted_class = 1 if predicted_score >= 0.5 else 0  # 阈值为0.5
            return predicted_class, predicted_score  # 返回类别和情感分数
    else:
        return None, None

def main():
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    directory_path = "EATD"
    features, labels = preprocess_directory(directory_path, processor)

    if features is None or labels is None:
        raise ValueError("未能正确提取特征或标签")

    # 处理标签: 高于53为消极情感，否则为积极情感
    labels_encoded = [1 if label > 53 else 0 for label in labels]

    dataset = create_dataset(features, labels_encoded)
    if dataset is None:
        raise ValueError("创建数据集失败。")

    features = dataset['input_values']
    labels = dataset['label']

    max_length = min(max(len(feature) for feature in features), 200000)
    print(f"最大特征长度: {max_length}")

    features_padded = pad_features(features, max_length)
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.float)

    tensor_dataset = TensorDataset(features_padded, labels_tensor)

    # 划分训练集和验证集
    train_size = int(0.8 * len(tensor_dataset))
    val_size = len(tensor_dataset) - train_size
    train_dataset, val_dataset = random_split(tensor_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = features_padded.shape[1]
    hidden_size = 64
    num_layers = 2

    # 实例化模型
    model = SentimentGRU(input_size, hidden_size, num_layers)
    criterion = nn.BCEWithLogitsLoss()  # 更新损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=30)

    # 保存模型
    model_name = 'sentiment_gru_model.pth'
    torch.save(model.state_dict(), model_name)

    # 加载模型
    model = load_model(model_name, input_size, hidden_size, num_layers)

    audio_path = 'Q1.wav'
    predicted_class, predicted_score = predict(audio_path, processor, model, max_length)
    print(f'Predicted class: {"Positive" if predicted_class == 0 else "Negative"}, Score: {predicted_score}')
if __name__ == "__main__":
    main()

