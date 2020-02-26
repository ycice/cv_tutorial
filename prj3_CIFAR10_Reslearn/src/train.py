import torch
from torch import optim, nn
from dataset import data_loader
from constants import LEARNING_RATE, MOMENTUM, EPOCH, device
from model import ResidualLearning, RLBlock


def train():
    train_set, test_set = data_loader()
    model: torch.nn.Module = ResidualLearning([3, 4, 6, 3], [64, 128, 256, 512])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for idx, (x_train, y_gt) in enumerate(train_set):
            model.train()
            x_train = x_train.to(device)
            y_gt = y_gt.to(device)

            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_gt)
            loss.backward()
            optimizer.step()

            if idx % 12 == 0:
                accuracy = int(sum((y_pred.argmax(dim=1) == y_gt).int())) / x_train.shape[0]
                print(f'{epoch}_{idx} train accuracy is {accuracy}')

            if idx % 125 == 0:
                model.eval()
                test_accuracy = 0
                for x_test, y_test_gt in test_set:
                    x_test = x_test.to(device)
                    y_test_gt = y_test_gt.to(device)
                    y_test_pred = model(x_test)
                    test_accuracy += int(sum((y_test_pred.argmax(dim=1) == y_test_gt).int()))

                test_accuracy = test_accuracy / len(test_set.dataset)
                print(f'test accuracy is {test_accuracy}')


if __name__ == '__main__':
    train()
