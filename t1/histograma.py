#!/usr/bin/python

import sys
import cv2
from glob import glob

NUM_CLASSES = 5
K = 2
HIST_SIZE = 64 

# Calcula o histograma de cada imagem já os classificando
# e retorna uma lista de histogramas e uma lista de classes
def calc_histogram(images: list) -> tuple[list, list]:
    hists = []
    classes = []
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        hist = [cv2.calcHist([img], [i], None, [HIST_SIZE], [0, 256]) for i in range(3)]
        hist = [cv2.normalize(hist, None) for hist in hist]
        hists.append(hist)
        classes.append(idx // NUM_CLASSES)

    return hists, classes

def choose_method(method: int):
    """
    Escolhe o método de comparação de histogramas
    Retorna uma tupla que contém:
        Função de comparação,
        Métrica,
        Se a métrica é maior
    """
    match method:
        case 1:
            return cv2.norm, cv2.NORM_L2, False
        case 2:
            return cv2.compareHist, cv2.HISTCMP_CORREL, True
        case 3:
            return cv2.compareHist, cv2.HISTCMP_CHISQR, False
        case 4:
            return cv2.compareHist, cv2.HISTCMP_INTERSECT, True
        case 5:
            return cv2.compareHist, cv2.HISTCMP_BHATTACHARYYA, False
        case _:
            print('Método inválido')
            exit(1)

# Classifica a classe de um histograma com base na classe dos K vizinhos
def classify(scores: list, classes: list, bigger: bool) -> int:
    classes = classes.copy()
    scores, classes = zip(*sorted(zip(scores, classes), reverse=bigger))

    neighbors = classes[:K]
    counts = [0] * NUM_CLASSES
    for neighbor in neighbors:
        counts[neighbor] += 1

    return counts.index(max(counts))

# Cria a matriz de confusão com base nos histogramas e classes
def make_conf_mtx_by_method(hists: list, classes: list, method: int):
    confusion_matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    similarity_function, metric, bigger = choose_method(method)

    # Pega os scores com base na função de similaridade escolhida
    for i in range(len(hists)):
        scores = []
        for j in range(len(hists)):
            if i != j:
                scs = [similarity_function(hists[i][k], hists[j][k], metric) for k in range(3)]
                average = sum(scs) / 3
                scores.append(average)

        label = classify(scores, classes, bigger)
        confusion_matrix[i // NUM_CLASSES][label] += 1

    return confusion_matrix

def main() -> None:
    method = int(sys.argv[1])
    images_path = sys.argv[2]
    images = sorted(glob(f"{images_path}/*"))

    hists, classes = calc_histogram(images)
    confusion_matrix = make_conf_mtx_by_method(hists, classes, method)
    accuracy = sum([confusion_matrix[i][i] for i in range(NUM_CLASSES)]) / len(images)

    print('Accuracy:', accuracy)
    for row in confusion_matrix:
        print(row)

if __name__ == "__main__":
    main()
