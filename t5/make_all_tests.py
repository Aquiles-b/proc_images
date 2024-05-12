import random
import numpy as np
import torch
import torchvision.transforms as transforms
import os

from generate_data import train_model, create_custom_model
from generate_data import create_LBP_csv_data, create_CNN_csv_data
from core import TextureClassifier, read_hists
from eval import evaluate_KNN, evaluate_model


# Treina um modelo customizado, salva e retorna o modelo treinado.
def train_custom_model(image_dim: tuple[int, int]) -> torch.nn.Module: 
    cnn1, cnn2, clf = create_custom_model(image_dim, 9)
    model = TextureClassifier()
    model.custom_model(cnn1, cnn2, clf)

    path_to_save = './data/texture_clf_custom.pth'

    if os.path.exists(path_to_save):
        model.load_model(path_to_save)
    else:
        train_model(model, image_dim, path_to_save, 0.001, 30)

    return model

# Faz o fine-tuning do modelo VGG16, salva e retorna o modelo treinado.
def fine_tune_VGG16(image_dim: tuple[int, int], freeze: bool) -> torch.nn.Module:
    model = TextureClassifier()
    model.VGG16(9, freeze)

    path_to_save = './data/texture_clf_vgg16.pth'

    if os.path.exists(path_to_save):
        model.load_model(path_to_save)
    else:
        train_model(model, image_dim, path_to_save, 0.001, 35)

    return model

# Gera os dados do modelo customizado
def get_model_data(model: torch.nn.Module, name: str,
                   transform: transforms.Compose) -> None:
    create_CNN_csv_data(name, model, transform)
    conf_mtx_pred = evaluate_model(model, 'test', transform, num_classes=9)
    train_hist = read_hists(f'./data/train_{name}_CNN.csv')
    test_hist = read_hists(f'./data/test_{name}_CNN.csv')
    conf_mtx_knn = evaluate_KNN(train_hist, test_hist, num_classes=9, knn=1)

    #PAREI AQUI
    

def get_custom_model_metrics() -> None:
    image_dim = (384, 384)
    custom_model = train_custom_model(image_dim)
    transform = transforms.Compose([
            transforms.CenterCrop((768, 768)),
            transforms.Resize(image_dim),
            transforms.ToTensor()])
    get_model_data(custom_model, 'custom', transform)

def get_VGG16_model_metrics(freeze: bool) -> None:
    image_dim = (224, 224)
    vgg16_model = fine_tune_VGG16(image_dim, freeze)
    transform = transforms.Compose([
            transforms.Resize(image_dim),
            transforms.ToTensor()])
    get_model_data(vgg16_model, 'vgg16', transform)

def main() -> None:
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    os.makedirs('./data', exist_ok=True)

    # Gera os dados LBP caso não existam
    if not os.path.exists('./data/train_LBP.csv'):
        create_LBP_csv_data((816, 612))

    # Gera os dados do modelo customizado
    get_custom_model_metrics()

    # Gera os dados do modelo VGG16
    get_VGG16_model_metrics(True)
    get_VGG16_model_metrics(False)






if __name__ == '__main__':
    main()
