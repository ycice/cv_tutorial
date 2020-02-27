import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
TRAIN_JSON = os.path.join(DATA_DIR, 'snli_1.0_train.jsonl')
TEST_JSON = os.path.join(DATA_DIR, 'snli_1.0_test.jsonl')
TRAIN_PICKLE = os.path.join(DATA_DIR, 'snli_train.pkl')
TEST_PICKLE = os.path.join(DATA_DIR, 'snli_test.pkl')

LABEL_DICT = {'neutral': 0, 'contradiction': 1, 'entailment': 2}

INPUT_DATA_LENGTH = 150
BATCH_SIZE = 16
EPOCH = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
