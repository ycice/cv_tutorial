from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from constants import DATA_DIR, BATCH_SIZE


def data_loader():
    transform = transforms.Compose(transforms=[transforms.ToTensor()])

    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)

    train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader
