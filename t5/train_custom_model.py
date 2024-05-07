from texture_classifier_cnn import TextureClassifier, TextureDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import torch

def main() -> None:
    np.random.seed(2024)
    torch.manual_seed(2024)

    image_dim = (2448, 3264)
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

    model = TextureClassifier('cpu')
    model.create_custom_model(image_dim)

    start = time.time()
    print('Começando treino:')
    model.train_model(train_loader, val_loader, 1e-3)
    print(f'{(time.time() - start)/60} minutos')


if __name__ == "__main__":
    main()
