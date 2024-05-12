import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import LBP_dir_hists, TextureDataset, write_histograms_csv 
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import numpy as np
from numpy import ndarray as NDArray


CURRENT_DIR = os.path.dirname(__file__)

def _two_cnn(model: torch.nn.Module, loader: DataLoader) -> list[NDArray]:
    fv = []
    for img, label in loader:
        x1 = model.cnn1(img)
        x2 = model.cnn2(img)

        tam_x1 = x1.size(1) * x1.size(2) * x1.size(3)
        tam_x2 = x2.size(1) * x2.size(2) * x2.size(3)

        x = torch.cat((x1.view(-1, tam_x1), x2.view(-1, tam_x2)), dim=1)
        x = x.detach().numpy()[0]
        x = np.concatenate((x, [label.item()]))
        fv.append(x)

    return fv

def _one_cnn(model: torch.nn.Module, loader: DataLoader) -> list[NDArray]:
    fv = []
    for img, label in loader:
        x = model.cnn1(img)
        tam_x1 = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, tam_x1)
        x = x.detach().numpy()[0]
        x = np.concatenate((x, [label.item()]))
        fv.append(x)

    return fv

def create_CNN_csv_data(name: str, model: torch.nn.Module, transform: transforms.Compose) -> None:
    data_path = f'{CURRENT_DIR}/../data'

    macroscopic0_dir = f'{CURRENT_DIR}/../macroscopic0'
    sub_sets = ['train', 'val', 'test']

    for sub_set in sub_sets:
        dataset = TextureDataset(f"{macroscopic0_dir}/{sub_set}", transform)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        if model.cnn2 is not None:
            fv = _two_cnn(model, test_loader)
        else:
            fv = _one_cnn(model, test_loader)

        write_histograms_csv(f"{data_path}/{sub_set}_{name}_CNN.csv", fv)

def create_LBP_csv_data(image_dim : tuple[int, int], img_gray: bool = True) -> None:
    macroscopic0_dir = f'{CURRENT_DIR}/../macroscopic0'
    data_path = f'{CURRENT_DIR}/../data'

    sub_sets = ['train', 'val', 'test']
    
    for sub_set in sub_sets:
        csv_name = f"{data_path}/{sub_set}_LBP.csv"
        hists = LBP_dir_hists(f'{macroscopic0_dir}/{sub_set}', image_dim, img_gray=img_gray)
        write_histograms_csv(csv_name, hists)
