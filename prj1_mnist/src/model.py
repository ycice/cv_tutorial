import torch
from torch import nn
import torch.nn.functional as F  # 학습할 필요없는 함수들(ex : RELU)


class ConvMnistModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=1)  # 3 * 3 행렬로 convolution
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)  # 채널이 16에서 32로 가는 것
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 2 * 2 중에 가장 최고값만 남기고 다 없애는 풀링
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 64)  # 채널이 64개로 들어나고 pooling 두번으로 픽셀은 7 * 7로
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x: torch.FloatTensor):  # pytorh에서는 forward를 쓰면 클래스 자체적으로 함수역할함
        conv1_out = F.relu(self.conv1(x))  # 16*28*28
        conv2_out = F.relu(self.conv2(conv1_out))  # 32*28*28
        pool1_out = self.pool1(conv2_out)  # 32*14*14
        conv3_out = F.relu(self.conv3(pool1_out))  # 32*14*14
        conv4_out = F.relu(self.conv4(conv3_out))  # 64*14*14
        pool2_out = self.pool1(conv4_out)  # 64*7*7, pool1을 그대로쓰면 그라디언트가 꼬임

        flatten = pool2_out.view((-1, 64 * 7 * 7))

        fc1_out = F.relu(self.fc1(flatten))
        fc2_out = F.relu(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)  # 마지막에는 relu를 하면 안됨

        return fc3_out


class LinearMnistModel(nn.Module):  # () 클래스 상속
    def __init__(self):
        super().__init__()  # 상속받은 부모클래스부터 정의해줘야됨

        self.linear1 = nn.Linear(784, 32)  # input과 output(숫자는 감으로)
        self.linear2 = nn.Linear(32, 64)  # 32에서 64로 가는 64*32 메트릭스 임의로 생산
        self.linear3 = nn.Linear(64, 16)
        self.linear4 = nn.Linear(16, 10)  # 최종 output은 클래스 10개에 대한 벡터(즉, 크기 10짜리 벡터)

    def forward(self, x: torch.FloatTensor):
        # input : N * 1 * 28 * 28  NCHW  -> N * 784

        x_flatten = x.view(size=(-1, 784))  # view는 reshape시켜줌. 그리고 -1은 shape에서 남은 차원의미
        linear_out1 = F.relu(self.linear1(x_flatten))
        linear_out2 = F.relu(self.linear2(linear_out1))
        linear_out3 = F.relu(self.linear3(linear_out2))
        linear_out4 = self.linear4(linear_out3)  # 마지막에서는 non-linear를 안취함

        return linear_out4


if __name__ == '__main__':
    random_img = torch.randn(size=(16, 1, 28, 28))  # size크기의 랜덤한 텐서 생산
    # model = LinearMnistModel()
    model = ConvMnistModel()
    random_y = model(random_img)
