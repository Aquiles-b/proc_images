from texture_classifier_cnn import TextureClassifier, TextureDataset
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import torch


def calc_input_mlp(image_dim: tuple[int, int], cnn: nn.Sequential) -> int:
    x = torch.randn(1, 3, *image_dim)
    y = cnn(x)
    return y.view(1, -1).shape[1]

def create_custom_model(image_dim: tuple[int, int], num_classes: int = 9) -> tuple[nn.Sequential, nn.Sequential]:
    cnn = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3,3)),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=(4,4), stride=4),

        nn.Conv2d(64, 128, kernel_size=(3,3)),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=(4,4), stride=4),

        nn.Conv2d(128, 128, kernel_size=(3,3)),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )

    fc1_input = calc_input_mlp(image_dim, cnn)

    classifier = nn.Sequential(
        nn.Linear(fc1_input, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 180),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(180, num_classes)
        )

    return cnn, classifier

def main() -> None:
    np.random.seed(2024)
    torch.manual_seed(2024)

    # image_dim = (2448, 3264)
    image_dim = (1224, 1632)
    transform = transforms.Compose([
        transforms.Resize(image_dim),
        transforms.ToTensor(),
        ])

    # Carrega os conjuntos de dados para treino
    train_data = TextureDataset('./macroscopic0/train', transform)
    val_data = TextureDataset('./macroscopic0/val', transform)

    batch_size = 2

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    cnn, clf = create_custom_model(image_dim, 9)

    model = TextureClassifier('cpu')
    model.custom_model(cnn, clf)

    start = time.time()
    print('Começando treino:')
    model.train_model(train_loader, val_loader, 1e-3, 20)
    print(f'{(time.time() - start)/60} minutos')


if __name__ == "__main__":
    main()
