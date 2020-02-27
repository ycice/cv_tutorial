import json
import pickle
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
from constants import TRAIN_JSON, TEST_JSON, TEST_PICKLE, TRAIN_PICKLE, INPUT_DATA_LENGTH, LABEL_DICT


class SNLI(Dataset):
    def __init__(self, input_list, segment_list, output_list):
        super().__init__()
        self.input_list = input_list
        self.segment_list = segment_list
        self.output_list = output_list

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        x_token_short = torch.LongTensor(self.input_list[index])
        x_zeros = torch.zeros(INPUT_DATA_LENGTH - x_token_short.shape[0]).type(torch.LongTensor)
        x_token = torch.cat([x_token_short, x_zeros])

        x_segment_short = torch.LongTensor(self.segment_list[index])
        x_zeros = torch.zeros(INPUT_DATA_LENGTH - x_segment_short.shape[0]).type(torch.LongTensor)
        x_segment = torch.cat([x_segment_short, x_zeros])

        y = torch.LongTensor(self.output_list[index])
        return x_token, x_segment, y


def dump_pickle(dataset, path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)


def read_jsonl(down_path, up_path):
    in_list = []
    out_list = []
    segment_list = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(down_path, 'r') as f:
        text = f.readlines()
        for line in text:
            json_line = json.loads(line)
            sentence1 = json_line['sentence1']
            sentence2 = json_line['sentence2']

            tokenized1 = tokenizer.encode(text=sentence1, add_special_tokens=False)
            tokenized2 = tokenizer.encode(text=sentence2, add_special_tokens=False)

            tokenized_sen = [101] + tokenized1 + [102] + tokenized2 + [102]

            # tokenized_sen = tokenizer.encode(text=sentence1, text_pair=sentence2, add_special_tokens=True)
            seg_zeros = torch.zeros(len(tokenized1) + 2, dtype=torch.long)
            seg_ones = torch.ones(len(tokenized2) + 1, dtype=torch.long)
            segment = torch.cat([seg_zeros, seg_ones])

            label = json_line['gold_label']
            if label not in LABEL_DICT:
                continue

            ans = LABEL_DICT[label]

            in_list.append(tokenized_sen)
            out_list.append([ans])
            segment_list.append(segment)
    snli_dataset = SNLI(in_list, segment_list, out_list)
    dump_pickle(snli_dataset, up_path)


if __name__ == '__main__':
    read_jsonl(TRAIN_JSON, TRAIN_PICKLE)
    read_jsonl(TEST_JSON, TEST_PICKLE)
