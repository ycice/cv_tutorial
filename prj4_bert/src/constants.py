import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
PICKLE_TRAIN_PATH = os.path.join(DATA_DIR, 'agnews.pkl')
PICKLE_TEST_PATH = os.path.join(DATA_DIR, 'agnews_test.pkl')
INPUT_DATA_LENGTH = 48
BATCH_SIZE = 16
EPOCH = 10
MOMENTUM = 0.9
LEARNING_RATE = 0.01
