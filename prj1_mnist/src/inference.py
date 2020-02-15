import os
import torch
from torchvision import transforms
from PIL import Image
from model import ConvMnistModel
from constants import DATASET_DIR, device
from util import load_model


def infer():
    sample_path = os.path.join(DATASET_DIR, 'mnist_sample.png')
    sample_pil: Image.Image = Image.open(sample_path)
    grayscale_pil = sample_pil.convert('L')
    resized_pil = grayscale_pil.resize(size=(28, 28))
    to_tensor = transforms.ToTensor()
    sample_tensor = to_tensor(resized_pil).unsqueeze(dim=0)  # 0차원에다가 1을 추가해서 차원수를 늘려줌

    sample_tensor = sample_tensor.to(device)

    model: torch.nn.Module = ConvMnistModel()
    load_model(model, f'conv_2_1000')
    model.to(device)

    model.eval()
    y_pred = model(sample_tensor)
    print(y_pred.argmax())


if __name__ == '__main__':
    infer()
