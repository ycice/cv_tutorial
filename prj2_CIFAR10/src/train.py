import torch
from torch import nn
from torch import optim
from util import save_model
from prj2_CIFAR10.src.model import DeepConvModel
from prj2_CIFAR10.src.dataloader import get_dataloader
from prj2_CIFAR10.src.constants import LEARNING_RATE, MOMENTUM, EPOCH


def train():
    train_loader, test_loader = get_dataloader()

    model = DeepConvModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=0.0001)

    for epoch in range(EPOCH):
        for step_index, (x_train, y_gt) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            y_pred: torch.FloatTensor = model(x_train)

            loss = criterion(y_pred, y_gt)
            loss.backward()
            optimizer.step()

            if step_index % 100 == 0:
                y_pred_argmax = y_pred.argmax(dim=1)
                y_pred_argmax_sum = int(sum((y_gt == y_pred_argmax).int()))
                train_accuracy = y_pred_argmax_sum / x_train.shape[0]

                train_top5_error = 0
                for i in range(x_train.shape[0]):
                    if y_gt[i] not in y_pred.topk(5).indices[i]:  # topk(j)는 가장 높은 확률을 j개까지 구해줌
                        train_top5_error += 1

                train_top5_error = train_top5_error / x_train.shape[0]
                print(f'{step_index}, {train_accuracy}, {loss}, {train_top5_error}')

            if step_index % 1000 == 0 and step_index > 0:
                model.eval()
                total_correct = 0
                top5_error = 0
                total_items = len(test_loader.dataset)
                for test_step, (x_test, y_test_gt) in enumerate(test_loader):
                    y_test_pred: torch.FloatTensor = model(x_test)
                    y_test_argmax = y_test_pred.argmax(dim=1)
                    total_correct += int(sum((y_test_gt == y_test_argmax).int()))

                    # top_5_error 구하기
                    for i in range(x_test.shape[0]):
                        if y_test_gt[i] not in y_test_pred.topk(5).indices[i]:  # topk(j)는 가장 높은 확률을 j개까지 구해줌
                            top5_error += 1

                top5_error = top5_error / total_items
                test_accuracy = total_correct / total_items
                print(f'test accuracy is {test_accuracy}, top5 error is {top5_error}')

                save_model(model, f'conv_{epoch}_{step_index}')


if __name__ == '__main__':
    train()
    # 8 layers-test accuracy : 0.716
    # 10 layers-test accuracy : 0.731
    # residual 13 layers : 0.712
    # residual 20 layers : 0.722
