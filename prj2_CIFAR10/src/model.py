import torch
from torch import nn
import torch.nn.functional as F


class DeepConvModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv7 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.fc1 = nn.Linear(512 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x: torch.FloatTensor):
        conv1_out = F.relu(self.conv1(x))  # 16*32*32
        conv2_out = F.relu(self.conv2(conv1_out))  # 32*32*32
        conv3_out = F.relu(self.conv3(conv2_out))  # 64*32*32
        pool1_out = self.pool1(conv3_out)  # 64*16*16
        conv4_out = F.relu(self.conv4(pool1_out))  # 128*16*16
        conv5_out = F.relu(self.conv5(conv4_out))  # 256*16*16
        conv6_out = F.relu(self.conv6(conv5_out))  # 512*16*16
        pool2_out = self.pool2(conv6_out)  # 512*8*8
        conv7_out = F.relu(self.conv7(pool2_out))  # 512*8*8

        flatten = conv7_out.view((-1, 512 * 8 * 8))

        fc1_out = F.relu(self.fc1(flatten))
        fc2_out = F.relu(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)

        return fc3_out


class ResidualLearning(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=1)  # 16*32*32
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.conv7 = nn.Conv2d(16, 16, (3, 3), padding=1)

        self.conv8 = nn.Conv2d(16, 32, (3, 3), padding=1, stride=2)  # 32*16*16
        self.conv9 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv10 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv11 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv13 = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv14 = nn.Conv2d(32, 64, (3, 3), padding=1, stride=2)  # 64*8*8
        self.conv15 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv16 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv17 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv18 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv19 = nn.Conv2d(64, 64, (3, 3), padding=1)

        self.fc = nn.Linear(64 * 8 * 8, 10)

        self.id1 = nn.Identity(54, unused_argument1=0.1)
        self.id2 = nn.Identity(54, unused_argument1=0.1)
        self.id3 = nn.Identity(54, unused_argument1=0.1)
        self.id4 = nn.Identity(54, unused_argument1=0.1)
        self.id5 = nn.Identity(54, unused_argument1=0.1)
        self.id6 = nn.Identity(54, unused_argument1=0.1)
        self.id7 = nn.Identity(54, unused_argument1=0.1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(16)
        self.bn7 = nn.BatchNorm2d(16)

        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(32)
        self.bn10 = nn.BatchNorm2d(32)
        self.bn11 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(32)
        self.bn13 = nn.BatchNorm2d(32)

        self.bn14 = nn.BatchNorm2d(64)
        self.bn15 = nn.BatchNorm2d(64)
        self.bn16 = nn.BatchNorm2d(64)
        self.bn17 = nn.BatchNorm2d(64)
        self.bn18 = nn.BatchNorm2d(64)
        self.bn19 = nn.BatchNorm2d(64)

    def forward(self, x: torch.FloatTensor):
        conv1_out = F.relu(self.bn1(self.conv1(x)))
        conv2_out = F.relu(self.bn2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.bn3(self.conv3(conv2_out) + self.id1(conv1_out)))
        conv4_out = F.relu(self.bn4(self.conv4(conv3_out)))
        conv5_out = F.relu(self.bn5(self.conv5(conv4_out) + self.id2(conv3_out)))
        conv6_out = F.relu(self.bn6(self.conv6(conv5_out)))
        conv7_out = F.relu(self.bn7(self.conv7(conv6_out) + self.id3(conv5_out)))

        conv8_out = F.relu(self.bn8(self.conv8(conv7_out)))
        conv9_out = F.relu(self.bn9(self.conv9(conv8_out)))
        conv10_out = F.relu(self.bn10(self.conv10(conv9_out)))
        conv11_out = F.relu(self.bn11(self.conv11(conv10_out) + self.id4(conv9_out)))
        conv12_out = F.relu(self.bn12(self.conv12(conv11_out)))
        conv13_out = F.relu(self.bn13(self.conv13(conv12_out) + self.id5(conv11_out)))

        conv14_out = F.relu(self.bn14(self.conv14(conv13_out)))
        conv15_out = F.relu(self.bn15(self.conv15(conv14_out)))
        conv16_out = F.relu(self.bn16(self.conv16(conv15_out)))
        conv17_out = F.relu(self.bn17(self.conv17(conv16_out) + self.id6(conv15_out)))
        conv18_out = F.relu(self.bn18(self.conv18(conv17_out)))
        conv19_out = F.relu(self.bn19(self.conv19(conv18_out) + self.id7(conv17_out)))

        flatten = conv19_out.view((-1, 64 * 8 * 8))
        fc_out = self.fc(flatten)

        return fc_out


if __name__ == '__main__':
    random_img = torch.randn(size=(16, 3, 32, 32))
    model = ResidualLearning()
    random_y = model(random_img)
