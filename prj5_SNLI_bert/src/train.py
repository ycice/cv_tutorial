import torch
import os
from torch import optim, nn
from pytorch_transformers import BertModel, AdamW
from dataloader import data_loader
from dataset import SNLI
from model import BertSnliFc
from constants import device, EPOCH, LEARNING_RATE, MOMENTUM, DATA_DIR


def save_model(model: torch.nn.Module, model_name: str):
    save_location = os.path.join(DATA_DIR, f'{model_name}.torch')
    torch.save(model.state_dict(), save_location)


origin_bert_model = BertModel.from_pretrained('bert-base-uncased')
fc_bert_model = BertSnliFc()
origin_bert_model.to(device)
fc_bert_model.to(device)

optimizer_fc = optim.SGD(fc_bert_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
optimizer_bert = AdamW(origin_bert_model.parameters(), lr=2e-5, eps=1e-8)

criterion = nn.CrossEntropyLoss()

train_set, test_set = data_loader()
for epoch in range(EPOCH):
    losses = []
    for index, (x_train, x_segment, y_train_gt) in enumerate(train_set):
        origin_bert_model.train()
        fc_bert_model.train()

        optimizer_fc.zero_grad()
        optimizer_bert.zero_grad()

        x_train = x_train.to(device)
        x_segment = x_segment.to(device)
        y_train_gt = (y_train_gt.T[0]).to(device)

        dummy_output = origin_bert_model(x_train, token_type_ids=x_segment)
        y_pred = fc_bert_model(dummy_output[1])
        loss = criterion(y_pred, y_train_gt)

        loss.backward()
        optimizer_bert.step()
        optimizer_fc.step()

        losses.append(loss)

        if index % 100 == 0:
            accuracy = int(sum((y_pred.argmax(dim=1) == y_train_gt).int())) / x_train.shape[0]
            print(f'{epoch}_{index} train accuracy is {accuracy}, loss is {sum(losses) / len(losses)}')

            losses = []

        if index % 1000 == 0:
            fc_bert_model.eval()
            origin_bert_model.eval()

            test_accuracy = 0
            for x_test, x_test_segment, y_test_gt in test_set:
                x_test = x_test.to(device)
                x_test_segment = x_test_segment.to(device)
                y_test_gt = (y_test_gt.T[0]).to(device)
                with torch.no_grad():
                    dummy_output = origin_bert_model(x_test, token_type_ids=x_test_segment)
                    y_test_pred = fc_bert_model(dummy_output[1])
                test_accuracy += int(sum((y_test_pred.argmax(dim=1) == y_test_gt).int()))

            test_accuracy = test_accuracy / len(test_set.dataset)
            print(f'test accuracy is {test_accuracy}')
            save_model(fc_bert_model, f'{epoch}_{index}_bert_fc_model')
            save_model(origin_bert_model, f'{epoch}_{index}_bert_origin_model')
