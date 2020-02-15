from torchvision import datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
from prj2_CIFAR10.src.constants import DATA_DIR, BATCH_SIZE


def get_dataloader():
    transformer = transforms.Compose(transforms=[transforms.ToTensor()])

    train_set = dset.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transformer)
    test_set = dset.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transformer)

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    x, y = get_dataloader()

