# load model from ../model/LeNet_MNIST.pth
# Build ui for user to input image and predict the number by gradio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image, ImageOps
import pandas as pd


mapping = pd.read_csv('../../data/emnist/emnist-balanced-mapping.txt', delimiter = ' ', index_col = 0, header = None)

# Define the model with Sequential
class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape((-1, 1, 28, 28))


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))

model = nn.Sequential(b1, b2, b3, b4,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(256, 47)).to("cuda:0")
model.load_state_dict(torch.load('../models/emnist-resnet-10-epochs.pth'))
model.eval()

# Define the function to predict the number
def predict_number(image):
    image = image.reshape(1, 1, 28, 28)
    image = torch.from_numpy(image).float().to("cuda:0")
    output = model(image)
    _, pred = torch.max(output, 1)
    return pred.item()

# 定义反色转换
class InvertColor:
    def __call__(self, img):
        return ImageOps.invert(img)

class ToBinary:
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        # 转换为灰度图像
        if img.mode != 'L':
            img = img.convert('L')
        # 应用二值化
        return img.point(lambda p: p > self.threshold and 255)

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小到28x28
    transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
    ToBinary(20),  # 二值化
    transforms.ToTensor(),  # 转换为Torch张量
])


# 处理图像的函数
def process_image(image):
    if isinstance(image, dict):  # 如果是手绘图像
        image = Image.fromarray(image['composite'])
    else:
        image = Image.fromarray(image)
    image = transform(image)
    return image


# 定义Gradio接口
def convert_and_predict(image):
    torch_image = process_image(image)
    numpy_image = torch_image.numpy()[0]  # 转换为numpy格式以便显示
    prediction = predict_number(numpy_image)
    prediction = f"{chr(int(mapping.loc[prediction].values))}"
    return numpy_image, prediction


with gr.Blocks() as demo:
    gr.Markdown("# ResNet EMNIST 手写字符识别")

    with gr.Row():
        with gr.Column():
            image_input = gr.ImageEditor(type="numpy", label="上传图像或手绘")
        with gr.Column():
            image_output = gr.Image(label="转换后的图像",height=448,width=448)
            image_result = gr.Label(label="预测结果：")

    convert_button = gr.Button("预测")
    convert_button.click(convert_and_predict, inputs=image_input, outputs=[image_output, image_result])

if __name__ == "__main__":
    demo.launch()


