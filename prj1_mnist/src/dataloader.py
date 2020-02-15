from torchvision import datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
from constants import DATASET_DIR, BATCH_SIZE


def get_dataloader():
    mnist_tranform = transforms.Compose(transforms=[transforms.ToTensor()])  # pillow를 tensor로 전환

    train_set = dset.MNIST(root=DATASET_DIR, train=True, download=True, transform=mnist_tranform)
    test_set = dset.MNIST(root=DATASET_DIR, train=False, download=True, transform=mnist_tranform)

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # shuffle은 랜덤으로 뽑는지
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    get_dataloader()
