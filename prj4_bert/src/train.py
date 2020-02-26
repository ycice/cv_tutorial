import torch
from torch import optim, nn
from pytorch_transformers import BertModel
from constants import device, EPOCH, LEARNING_RATE, MOMENTUM
from dataloader import dataloader
from dataset import AgNews  # class를 사용하기 위해서는 불러와야됨
from model import BertFc

origin_bert_model = BertModel.from_pretrained('bert-base-uncased')  # Bert는 LongTensor를 input으로 받는다
bert_fc_model: torch.nn.Module = BertFc()
origin_bert_model.to(device)
bert_fc_model.to(device)

optimizer = optim.SGD(bert_fc_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()
train_set, test_set = dataloader()

for epoch in range(EPOCH):
    for index, (x_train, y_trian_gt) in enumerate(train_set):
        bert_fc_model.train()

        x_train = x_train.to(device)
        y_trian_gt = (y_trian_gt.T[0] - 1).to(device)  # 클래스값이 0부터 시작해야되므로 1씩 빼준다.
        # 강제로 dim을 맞췄는데...?

        with torch.no_grad():
            output = origin_bert_model(x_train)
            cls = output[1]

        optimizer.zero_grad()
        y_pred = bert_fc_model(cls)
        loss = criterion(y_pred, y_trian_gt)
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            accuracy = int(sum((y_pred.argmax(dim=1) == y_trian_gt).int())) / x_train.shape[0]
            print(f'{epoch}_{index} train accuracy is {accuracy}')

        if index % 1000 == 0:
            bert_fc_model.eval()
            test_accuracy = 0
            for x_test, y_test_gt in test_set:
                x_test = x_test.to(device)
                y_test_gt = (y_test_gt.T[0] - 1).to(device)
                with torch.no_grad():
                    out = origin_bert_model(x_test)
                y_test_pred = bert_fc_model(out[1])
                test_accuracy += int(sum((y_test_pred.argmax(dim=1) == y_test_gt).int()))

            test_accuracy = test_accuracy / len(test_set.dataset)
            print(f'test accuracy is {test_accuracy}')
