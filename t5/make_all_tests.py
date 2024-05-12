import random
import numpy as np
import torch
import torchvision.transforms as transforms
import os

from generate_data import train_model, create_custom_model
from generate_data import create_LBP_csv_data, create_CNN_csv_data
from core import TextureClassifier, read_lists_csv, write_lists_csv, LBPDataset
from eval import evaluate_KNN, evaluate_model, evaluate_clf_LBP
from torch.utils.data import DataLoader


def train_LBP_classifier(num_epochs: int, lr: float) -> torch.nn.Module:
    batch_size = 32

    train_hists = read_lists_csv('./data/train_LBP.csv')
    val_hists = read_lists_csv('./data/val_LBP.csv')
    train_dataset = LBPDataset(train_hists)
    val_dataset = LBPDataset(val_hists)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TextureClassifier()
    model.create_LBP_clf(58, 9)

    path_to_save = './data/texture_clf_LBP.pth'
    if os.path.exists(path_to_save):
        model.load_model(path_to_save)
    else:
        model.train_model(train_loader, val_loader, lr, num_epochs, path_to_save)

    return model

# Treina um modelo customizado, salva e retorna o modelo treinado.
def train_custom_model(image_dim: tuple[int, int]) -> torch.nn.Module: 
    cnn1, cnn2, clf = create_custom_model(image_dim, 9)
    model = TextureClassifier()
    model.custom_model(cnn1, cnn2, clf)

    path_to_save = './data/texture_clf_custom.pth'

    if os.path.exists(path_to_save):
        model.load_model(path_to_save)
    else:
        train_model(model, image_dim, path_to_save, 0.01, 15)
        train_model(model, image_dim, path_to_save, 0.001, 30)

    return model

# Faz o fine-tuning do modelo VGG16, salva e retorna o modelo treinado.
def fine_tune_VGG16(image_dim: tuple[int, int], freeze: bool) -> torch.nn.Module:
    model = TextureClassifier()
    model.VGG16(9, freeze)

    if freeze:
        path_to_save = './data/texture_clf_vgg16_freezed.pth'
    else:
        path_to_save = './data/texture_clf_vgg16_finetuned.pth'

    if os.path.exists(path_to_save):
        model.load_model(path_to_save)
    else:
        train_model(model, image_dim, path_to_save, 0.001, 35)

    return model

# Retorna a acurácia do modelo CNN classificado com o próprio modelo e com o KNN.
def get_model_metrics(model: torch.nn.Module, name: str,
                   transform: transforms.Compose) -> tuple[float, float]:
    create_CNN_csv_data(name, model, transform)
    conf_mtx_pred = evaluate_model(model, 'test', transform, num_classes=9)
    train_hist = read_lists_csv(f'./data/train_{name}_CNN.csv')
    test_hist = read_lists_csv(f'./data/test_{name}_CNN.csv')
    conf_mtx_knn = evaluate_KNN(train_hist, test_hist, num_classes=9, knn=1)

    accuracy_pred = np.trace(conf_mtx_pred) / np.sum(conf_mtx_pred)
    accuracy_knn = np.trace(conf_mtx_knn) / np.sum(conf_mtx_knn)

    return accuracy_pred, accuracy_knn
    
def get_custom_model_metrics() -> list:
    image_dim = (384, 384)
    custom_model = train_custom_model(image_dim)
    transform = transforms.Compose([
            transforms.CenterCrop((768, 768)),
            transforms.Resize(image_dim),
            transforms.ToTensor()])
    acc_pred, acc_knn = get_model_metrics(custom_model, 'custom', transform)

    return [acc_pred, acc_knn]

def get_VGG16_model_metrics(freeze: bool) -> list:
    image_dim = (224, 224)
    vgg16_model = fine_tune_VGG16(image_dim, freeze)
    transform = transforms.Compose([
            transforms.Resize(image_dim),
            transforms.ToTensor()])
    acc_pred, acc_knn = get_model_metrics(vgg16_model, 'vgg16', transform)

    return [acc_pred, acc_knn]

def get_LBP_metrics(clf: torch.nn.Module) -> tuple[float, float]:
    train_hist = read_lists_csv('./data/train_LBP.csv')
    test_hist = read_lists_csv('./data/test_LBP.csv')
    conf_mtx_knn = evaluate_KNN(train_hist, test_hist, num_classes=9, knn=1)
    conf_mtx_pred = evaluate_clf_LBP(clf, 'test', 9)

    accuracy_knn = np.trace(conf_mtx_knn) / np.sum(conf_mtx_knn)
    accuracy_pred = np.trace(conf_mtx_pred) / np.sum(conf_mtx_pred)

    return accuracy_pred, accuracy_knn

def main() -> None:
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    os.makedirs('./data', exist_ok=True)

    performance_table = []
    performance_table.append(['Representação', 'Classificador (MLP)', 'KNN'])

    # Gera os dados LBP 
    create_LBP_csv_data((816, 612))
    clf = train_LBP_classifier(100, lr=0.001)
    acc_pred, acc_knn = get_LBP_metrics(clf)
    performance_table.append(['LBP', acc_pred, acc_knn])

    # Gera os dados do modelo customizado
    acc_p, acc_k = get_custom_model_metrics()
    performance_table.append(['Custom model', acc_p, acc_k])

    # Gera os dados do modelo VGG16
    acc_p, acc_k = get_VGG16_model_metrics(True)
    performance_table.append(['VGG16 (frozen)', acc_p, acc_k])
    acc_p, acc_k = get_VGG16_model_metrics(False)
    performance_table.append(['VGG16 (fine-tuned)', acc_p, acc_k])

    write_lists_csv('./performance_table.csv', performance_table)


if __name__ == '__main__':
    main()
