import torch
from torch import nn
from torch import optim

from dataset import get_dataloader
from util import save_model
from model import LinearMnistModel, ConvMnistModel
from constants import LEARNING_RATE, NUM_EPOCH, device


def train():
    train_loader, test_loader = get_dataloader()

    # model = LinearMnistModel()
    model = ConvMnistModel()

    model.to(device)  # gpu가 있으면 gpu를 쓰고 아니면 cpu를 씀

    criterion = nn.CrossEntropyLoss()  # loss fct
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch_index in range(NUM_EPOCH):  # train set을 몇번 돌릴것인지
        for step_index, (x_train, y_gt) in enumerate(train_loader):  # enumerate에는 순서도 같이 나옴
            model.train()  # train mode로 들어감을 알려줌
            optimizer.zero_grad()  # grad를 0으로 해줌(안하면 누적됨)

            x_train = x_train.to(device)
            y_gt = y_gt.to(device)

            y_pred: torch.FloatTensor = model(x_train)

            loss = criterion(y_pred, y_gt)
            loss.backward()
            optimizer.step()

            if step_index % 100 == 0:
                y_pred_argmax = y_pred.argmax(dim=1)  # 몇번째 dim에서 계산할것인지(0번째 dim은 batch이므로)
                num_correct = int(sum((y_gt == y_pred_argmax).int()))
                train_accuracy = num_correct / x_train.shape[0]
                print(loss, train_accuracy)

            if step_index % 1000 == 0:
                model.eval()  # drop_out이랑 batch_normal 꺼짐. 평가모드로 바뀜
                total_correct = 0
                total_items = len(test_loader.dataset)  # test셋의 총 개수
                for text_step_index, (x_test, y_test_gt) in enumerate(test_loader):
                    x_test = x_test.to(device)
                    y_test_gt = y_test_gt.to(device)

                    y_pred: torch.FloatTensor = model(x_test)
                    y_pred_argmax = y_pred.argmax(dim=1)

                    num_correct = int(sum((y_test_gt == y_pred_argmax).int()))
                    total_correct += num_correct

                test_accuracy = total_correct / total_items
                print(f'test accuracy is {test_accuracy}')

                save_model(model, f'conv_{epoch_index}_{step_index}')


if __name__ == '__main__':
    train()
    # text accuracy is 0.956
