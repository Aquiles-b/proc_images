import numpy as np
from numpy._typing import NDArray


# @test é o feature vector a ser classificado.
# @training_list é a lista de features vectors de treinamento.
def KNN_decision(test: NDArray, training_list: list[NDArray], 
                 knn: int = 1, num_classes: int = 9, use_cos_distance: bool = True) -> int:
    if use_cos_distance:
        distances = [cos_similarity(test[:-1], hist[:-1]) for hist in training_list]
        k_nearest = np.argsort(distances)[-knn:]
    else:
        distances = [euclidean_distance(test[:-1], hist[:-1]) for hist in training_list]
        k_nearest = np.argsort(distances)[:knn]

    voting_list = [0] * num_classes
    for i in range(knn):
        voting_list[i] += training_list[k_nearest[i]][-1]
    idx = np.argmax(voting_list)

    return int(voting_list[idx])

# Calcula a distancia euclidiana entre 2 pontos.
def euclidean_distance(x1: NDArray, x2: NDArray) -> float:
    return np.sqrt(np.sum((x1-x2)**2))

def cos_similarity(x1: NDArray, x2: NDArray) -> float:
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
