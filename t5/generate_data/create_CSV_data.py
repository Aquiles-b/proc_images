import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import LBP_dir_hists, TextureDataset, TextureClassifier, write_histograms_csv 
from generate_data import create_custom_model
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

def create_CNN_csv_data(model: torch.nn.Module) -> None:
    data_path = f'{CURRENT_DIR}/../data'
    os.makedirs(data_path, exist_ok=True)

    dim = (384, 384)

    transform = transforms.Compose([
            transforms.CenterCrop((768, 768)),
            transforms.Resize(dim),
            transforms.ToTensor(),
        ])

    macroscopic0_dir = f'{CURRENT_DIR}/../macroscopic0'
    sub_sets = ['train', 'val', 'test']

    for sub_set in sub_sets:
        dataset = TextureDataset(f"{macroscopic0_dir}/{sub_set}", transform)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        if model.cnn2 is not None:
            fv = _two_cnn(model, test_loader)
        else:
            fv = _one_cnn(model, test_loader)

        write_histograms_csv(f"{data_path}/{sub_set}_CNN.csv", fv)

def create_LBP_csv_data(img_gray: bool = True) -> None:
    macroscopic0_dir = f'{CURRENT_DIR}/../macroscopic0'
    data_path = f'{CURRENT_DIR}/../data'

    os.makedirs(data_path, exist_ok=True)

    image_dim = (3264, 2448)
    fator = 2
    image_dim = (image_dim[0] // fator, image_dim[1] // fator)

    sub_sets = ['train', 'val', 'test']
    
    for sub_set in sub_sets:
        hists = LBP_dir_hists(f'{macroscopic0_dir}/{sub_set}', image_dim, img_gray=img_gray)
        write_histograms_csv(f"{data_path}/{sub_set}_LBP.csv", hists)

if __name__ == '__main__':
    # model = TextureClassifier('cpu')
    # model.custom_model(*create_custom_model((384, 384), 9))
    model = TextureClassifier('cpu')
    model.resnet50(9, freeze=True)
    # create_LBP_csv_data(False)
    create_CNN_csv_data(model)
