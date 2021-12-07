import torch
import torchvision.datasets as da
import torchvision.transforms as tf

def get_trainloader(da_name, path, img_size, batch_size):
    transform = tf.Compose([
        tf.Resize(img_size),
        tf.ToTensor(),
    ])

    if da_name == "CIFAR":
        dataset = da.CIFAR10(root=path, train=True, transform=transform, download=True)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)


def get_testloader(da_name, path, img_size, batch_size = 1):
    transform = tf.Compose([
        tf.Resize(img_size),
        tf.ToTensor(),
    ])

    if da_name == "CIFAR":
        dataset = da.CIFAR10(root=path, train=False, transform=transform, download=True)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.classes)