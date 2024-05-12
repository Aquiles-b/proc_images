import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import TextureDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(__file__)

def evaluate_model(model: torch.nn.Module, sub_set: str, 
                   transform: transforms.Compose, num_classes: int = 9):
    dataset = TextureDataset(f"{CURRENT_DIR}/../macroscopic0/{sub_set}", transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    for img, lbl in test_loader:
        out = model.predict(img)
        confusion_matrix[lbl][out["class"]] += 1

    return confusion_matrix
