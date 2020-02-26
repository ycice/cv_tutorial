import csv
import pickle
from typing import List  # list의 타입을 더 자세하게 정의해줌
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
from constants import TRAIN_PATH, PICKLE_TRAIN_PATH, INPUT_DATA_LENGTH, PICKLE_TEST_PATH, TEST_PATH
from tqdm import tqdm


# dataset 직접 만들기
class AgNews(Dataset):
    def __init__(self, input_texts: List[List[int]], output_class: List[List[int]]):
        self.x_data: List[List[int]] = input_texts
        self.y_data = output_class

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_short = torch.LongTensor(self.x_data[idx])
        zeros_list = torch.zeros(INPUT_DATA_LENGTH - x_short.shape[0]).type(torch.LongTensor)
        # torch.zeros은 numpy.zeros와 같은 역할. 이때 형태는 FloatTensor로 나옴.
        # .type(torch.LongTensor)하면 FloatTensor를 LongTensor로 변환해줌
        # FloatTensor는 32bit, LongTensor는 64bit

        x = torch.cat([x_short, zeros_list])  # 나머지 차원이 같은 때, 두 행렬을 붙여줌
        y = torch.LongTensor(self.y_data[idx])
        return x, y


def pickle_dump(arg_dataset, path):
    with open(path, 'wb') as f:
        pickle.dump(arg_dataset, f)


def build_dataset(path):
    input_lists = []
    output_lists = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(path, 'r') as f:
        text = csv.reader(f)
        for row in tqdm(text):
            tokenized_row = tokenizer.encode(row[1], add_special_tokens=True)
            # zero_list = torch.zeros(INPUT_DATA_LENGTH - len(tokenized_row))
            # tokenized_row = torch.cat([tokenized_row, zero_list])
            input_lists.append(tokenized_row)
            output_lists.append([int(row[0])])

    arg_dataset = AgNews(input_lists, output_lists)
    return arg_dataset


if __name__ == '__main__':
    arg_train = build_dataset(TRAIN_PATH)
    arg_test = build_dataset(TEST_PATH)
    pickle_dump(arg_train, PICKLE_TRAIN_PATH)
    pickle_dump(arg_test, PICKLE_TEST_PATH)
