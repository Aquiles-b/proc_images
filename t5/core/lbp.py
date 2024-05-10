import numpy as np
from numpy._typing import NDArray
from skimage import feature
from PIL import Image
from glob import glob


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

# Gerar histogramas LBP das images no diretorio @imgs_dir e retorna
# uma lista de histogramas.
def LBP_dir_hists(imgs_dir: str, image_dim: tuple,
                  img_gray: bool = True) -> list[NDArray]:
    imgs = glob(f'{imgs_dir}/*')

    if img_gray:
        lbp_func = LBP_feature_hist_gray
        mode = 'L'
    else:
        lbp_func = LBP_feature_hist
        mode = 'RGB'

    hist_list = list()

    for i in imgs:
        label = int(i.split('/')[-1][:2]) - 1
        img = Image.open(i).convert(mode)
        img = img.resize(image_dim)
        hist = lbp_func(img, label)
        hist_list.append(hist)

    return hist_list

