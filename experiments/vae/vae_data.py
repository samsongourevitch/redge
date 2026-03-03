from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_vae_train_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),  # binarize
        transforms.Lambda(lambda x: x.view(-1)),         # flatten
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
