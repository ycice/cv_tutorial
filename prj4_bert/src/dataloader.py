import pickle
from torch.utils.data import DataLoader
from constants import PICKLE_TRAIN_PATH, BATCH_SIZE, PICKLE_TEST_PATH
from dataset import AgNews


def pickle_load(path):
    with open(path, 'rb') as f:
        pickle_text: AgNews = pickle.load(f)
    return pickle_text


def dataloader():
    arg_train_dataset = pickle_load(PICKLE_TRAIN_PATH)
    arg_test_dataset = pickle_load(PICKLE_TEST_PATH)
    train_dataloader = DataLoader(dataset=arg_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=arg_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_set = dataloader()
    for x, y in train_set:
        print(1)
