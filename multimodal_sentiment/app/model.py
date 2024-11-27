"""
File: model.py
Author: Elena Ryumina and Dmitry Ryumin
Description: This module provides functions for loading and processing a pre-trained deep learning model
             for facial expression recognition.
License: MIT License
"""
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM

# Importing necessary components for the Gradio app
from app.config import config_data
from app.model_architectures import ResNet50, LSTMPyTorch

# 直接从本地加载模型
def load_model(model_path):
    try:
        # 检查模型文件是否存在
        if model_path:
            return model_path
        else:
            print(f"Model path not found: {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# 使用本地路径来加载静态模型
path_static = load_model(config_data.model_static_path)
pth_model_static = ResNet50(7, channels=3)
pth_model_static.load_state_dict(torch.load(path_static))
pth_model_static.eval()

# 使用本地路径来加载动态模型
path_dynamic = load_model(config_data.model_dynamic_path)
pth_model_dynamic = LSTMPyTorch()
pth_model_dynamic.load_state_dict(torch.load(path_dynamic))
pth_model_dynamic.eval()

# 设置 GradCAM 使用的目标层
target_layers = [pth_model_static.layer4]
cam = GradCAM(model=pth_model_static, target_layers=target_layers)

# 图像预处理函数
def pth_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def __init__(self):
            super(PreprocessInput, self).__init__()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img, target_size=(224, 224)):
        transform = transforms.Compose([transforms.PILToTensor(), PreprocessInput()])
        img = img.resize(target_size, Image.Resampling.NEAREST)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        return img

    return get_img_torch(fp)

