import os
import torch
import librosa
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Processor

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
        self.dropout = nn.Dropout(0.5)  # 添加dropout层
        self.fc = nn.Linear(hidden_size, 1)  # 输出单个值

    def forward(self, x):
    #    print(f"Input tensor shape: {x.shape}")
     #   print(f"Input tensor dtype: {x.dtype}")

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])  # 添加dropout层
        out = self.fc(out)
        return out

def load_model(model_path, input_size, hidden_size, num_layers):
    model = SentimentGRU(input_size, hidden_size, num_layers)
    # 使用 CPU 加载模型
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(audio_path, processor, model, max_length):
    feature, _ = load_and_extract_features(audio_path, processor)
    if feature is not None:
        feature_padded = pad_features([feature], max_length)
        feature_tensor = feature_padded.unsqueeze(0)  # Shape: (1, 1, 200000)
        with torch.no_grad():
            output = model(feature_tensor)
            predicted_score = torch.sigmoid(output).item()
            predicted_class = 1 if predicted_score >= 0.5 else 0
            return predicted_class, predicted_score
    else:
        return None, None

def voice_sentiment(voice):
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    model_path = 'sentiment_gru_model.pth'  # 已训练好的模型路径
    input_size = 200000  # 根据最大特征长度设定
    hidden_size = 64
    num_layers = 2

    # 加载模型
    model = load_model(model_path, input_size, hidden_size, num_layers)

    # 需要分析的音频文件
    audio_path = voice

    # 进行情感预测
    max_length = input_size
    predicted_class, predicted_score = predict(audio_path, processor, model, max_length)
    print("预测结果:predicted_class, predicted_score", predicted_class, predicted_score)
    if predicted_class is not None:
        if predicted_class == 1:
            return f"积极", predicted_score
        else:
            return f"消极", 1. - predicted_score
    else:
        return "无法进行预测，请检查音频文件。"

