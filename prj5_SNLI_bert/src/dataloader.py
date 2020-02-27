import pickle
from torch.utils.data import DataLoader
from dataset import SNLI
from constants import TEST_PICKLE, TRAIN_PICKLE, BATCH_SIZE


def load_pickle(path):
    with open(path, 'rb') as f:
        pickle_text: SNLI = pickle.load(f)
    return pickle_text


def data_loader():
    train_data = load_pickle(TRAIN_PICKLE)
    test_data = load_pickle(TEST_PICKLE)

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader

