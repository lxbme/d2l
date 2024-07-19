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


# Define the model with Sequential
class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape((-1, 1, 28, 28))

model = nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6 ,kernel_size=5, padding=2),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(400, 120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10)
)


# Load the model
model.load_state_dict(torch.load('../models/LeNet_MNIST.pth'))
model.eval()

# Define the function to predict the number
def predict_number(image):
    image = image.reshape(1, 1, 28, 28)
    image = torch.from_numpy(image).float()
    output = model(image)
    _, pred = torch.max(output, 1)
    return pred.item()

# 定义反色转换
class InvertColor:
    def __call__(self, img):
        return ImageOps.invert(img)

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小到28x28
    transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
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
    prediction = f"预测结果：{prediction}"
    return numpy_image, prediction


with gr.Blocks() as demo:
    gr.Markdown("# LeNet MNIST 手写数字识别")

    with gr.Row():
        with gr.Column():
            image_input = gr.ImageEditor(type="numpy", label="上传图像或手绘")
        with gr.Column():
            image_output = gr.Image(label="转换后的图像",height=512,width=512)
            image_result = gr.Label(label="预测结果：")

    convert_button = gr.Button("预测")
    convert_button.click(convert_and_predict, inputs=image_input, outputs=[image_output, image_result])

if __name__ == "__main__":
    demo.launch()


