from texture_classifier_cnn import TextureClassifier, TextureDataset
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import torch


def calc_out_dim_model(x: int) -> int:
    for _ in range(3):
        x -= 2
        x = x // 4
    x -= 2
    x = x // 2
    return x

def create_custom_model(image_dimension: tuple[int, int]) -> tuple[nn.Sequential, nn.Sequential]:
        h, w = image_dimension
        cnn = nn.Sequential(
                nn.Conv2d(3, 64, (3,3)),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(4,4), stride=4),
                nn.Conv2d(64, 32, (3,3)),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(4,4), stride=4),
                nn.Conv2d(32, 32, (3,3)),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(4,4), stride=4),
                nn.Conv2d(32, 16, (3,3)),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(2,2), stride=2),
                nn.Flatten()
                )
        fc1_tam = calc_out_dim_model(w) * calc_out_dim_model(h) * 16
        classifier = nn.Sequential(
                nn.Linear(fc1_tam, 180),
                nn.Linear(180, 45),
                nn.Linear(45, 9)
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

    batch_size = 8

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    cnn, clf = create_custom_model(image_dim)

    model = TextureClassifier('cpu')
    model.custom_model(cnn, clf)

    start = time.time()
    print('Começando treino:')
    model.train_model(train_loader, val_loader, 2e-2, 20)
    print(f'{(time.time() - start)/60} minutos')


if __name__ == "__main__":
    main()
