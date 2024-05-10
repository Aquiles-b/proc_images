import numpy as np
from numpy._typing import NDArray
from skimage import feature
from PIL import Image
from glob import glob
import csv


def LBP_feature_hist(img: Image.Image, label: int) -> NDArray:
    img_a = np.array(img)

    fv = np.array([])
    for i in range(3):
        lbp = feature.local_binary_pattern(img_a[:,:, i], 8, 3, method='default')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 255))
        hist = normalize_hist(hist)
        fv = np.concatenate((fv, hist))
    fv = np.append(fv, label)

    return fv

# Retorna um historama normalizado vindo de um processamento LBP da imagem passada.
def LBP_feature_hist_gray(img: Image.Image, label: int) -> NDArray:
    img_a = np.array(img)
    lbp = feature.local_binary_pattern(img_a, 8, 3, method='default')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 255))
    hist = normalize_hist(hist)
    fv = np.append(hist, label)

    return fv

# Normaliza o histograma usando o metodo min-max.
def normalize_hist(hist: NDArray) -> list:
    hist_l = list(hist)
    sorted_index = np.argsort(hist_l)
    idx_max, idx_min = sorted_index[-1], sorted_index[0]
    max, min = hist_l[idx_max], hist_l[idx_min]
    div = max - min
    for idx, num_field in enumerate(hist):
        hist_l[idx] = (num_field - min) / div
    return hist_l

# Escreve a lista de histogramas no final do arquivo csv.
def write_histograms_csv(csv_name: str, hist_list: list) -> None:
    with open(csv_name, mode='a', newline=None) as hist_csv:
        writer = csv.writer(hist_csv, delimiter=';')
        writer.writerows(hist_list)

# Escreve em um csv o vetor de caracteristicas de cada imagem em @imgs_dir
# dentro do diretorio imgs_dir.
# image_dim: (width, height)
def create_LBP_csv_hists(imgs_dir: str, csv_name: str, 
                         image_dim: tuple, mode = 'L') -> None:
    imgs = glob(f'{imgs_dir}/*.JPG')

    if mode == 'L':
        lbp_func = LBP_feature_hist_gray
    else:
        lbp_func = LBP_feature_hist

    hist_list = list()

    for i in imgs:
        label = int(i.split('/')[-1][:2]) - 1
        img = Image.open(i).convert(mode)
        img = img.resize(image_dim)
        hist = lbp_func(img, label)
        hist_list.append(hist)

    write_histograms_csv(csv_name, hist_list)

# Retorna uma lista de listas de floats vindos do csv passado.
def read_LBP_hists(csv_name: str) -> list[NDArray]:
    with open(csv_name, 'r') as file:
        file_csv = csv.reader(file, delimiter=';')
        file_list = list()
        for row in file_csv:
            file_list.append(np.array([float(i) for i in row]))
        return file_list
