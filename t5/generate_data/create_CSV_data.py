import __basic_import__
from core import LBP_dir_hists, TextureDataset, TextureClassifier, write_histograms_csv 
from generate_data import create_custom_model
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)

def create_CNN_csv_data(device: str, model_weights: str = '') -> None:
    model = TextureClassifier(device)
    model.custom_model(*create_custom_model((384, 384), 9))

    data_dir = f'{CURRENT_DIR}/../data'
    if len(model_weights) != 0:
        model.load_model(model_weights)
    model.load_model(f"{data_dir}/texture_classifier384-384.pt")
    transform = transforms.Compose([
            transforms.CenterCrop((768, 768)),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])

    macroscopic0_dir = f'{CURRENT_DIR}/../macroscopic0'
    sub_sets = ['train', 'val', 'test']

    for sub_set in sub_sets:
        dataset = TextureDataset(f"{macroscopic0_dir}/{sub_set}", transform)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        fv = []
        for img, label in test_loader:
            x1 = model.cnn1(img)
            x2 = model.cnn2(img)

            tam_x1 = x1.size(1) * x1.size(2) * x1.size(3)
            tam_x2 = x2.size(1) * x2.size(2) * x2.size(3)

            x = torch.cat((x1.view(-1, tam_x1), x2.view(-1, tam_x2)), dim=1)
            x = x.detach().numpy()[0]
            x = np.concatenate((x, [label.item()]))
            fv.append(x)
        write_histograms_csv(f"{data_dir}/{sub_set}_CNN.csv", fv)

def create_LBP_csv_data(img_gray: bool = True) -> None:
    macroscopic0_dir = f'{CURRENT_DIR}/../macroscopic0'
    data_dir = f'{CURRENT_DIR}/../data'

    image_dim = (3264, 2448)
    fator = 4
    image_dim = (image_dim[0] // fator, image_dim[1] // fator)

    sub_sets = ['train', 'val', 'test']
    
    for sub_set in sub_sets:
        hists = LBP_dir_hists(f'{macroscopic0_dir}/{sub_set}', image_dim, img_gray=img_gray)

        write_histograms_csv(f"{data_dir}/{sub_set}_CNN.csv", hists)

if __name__ == '__main__':
    create_LBP_csv_data(False)
    create_CNN_csv_data('cpu')
