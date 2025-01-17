import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import TextureDataset
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time


CURRENT_DIR = os.path.dirname(__file__)

def create_custom_model() -> tuple[nn.Sequential, nn.Sequential]:
    cnn1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=(2,2), stride=2),

            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=(4,4), stride=4),

            nn.Conv2d(64, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=(2,2), stride=2)
            )

    cnn2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=(2,2), stride=2),

            nn.Conv2d(64, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=(2,2), stride=2),
            )

    return cnn1, cnn2

def train_model(model: nn.Module, image_dim: tuple[int, int],
                path_to_save: str, lr: float, epochs: int) -> None:

    transform = transforms.Compose([
        transforms.RandomRotation(360),
        transforms.RandomCrop((768, 768)),
        transforms.Resize(image_dim),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        ])

    macroscopic0_path = f'{CURRENT_DIR}/../macroscopic0'

    # Carrega os conjuntos de dados para treino
    train_data = TextureDataset(f'{macroscopic0_path}/train', transform)
    val_data = TextureDataset(f'{macroscopic0_path}/val', transform)

    batch_size = 16

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    start = time.time()
    print('Começando treino:')
    model.train_model(train_loader, val_loader, lr, epochs, path_to_save=path_to_save)
    print(f'{(time.time() - start)/60} minutos')
