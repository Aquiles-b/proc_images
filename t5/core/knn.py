import numpy as np
from numpy._typing import NDArray


# @test é o feature vector a ser classificado.
# @training_list é a lista de features vectors de treinamento.
def KNN_decision(test: NDArray, training_list: NDArray, 
                 knn: int = 1, num_classes: int = 9) -> int:
    # No calculo das distancia precisa do slicing :-1 para remover o label no final.
    distances = [euclidean_distance(test[:-1], hist[:-1]) for hist in training_list]
    # Pega o indice dos K mais proximos.
    k_nearest = np.argsort(distances)[:knn]

    voting_list = [0] * num_classes
    for i in range(knn):
        voting_list[i] += training_list[k_nearest[i]][-1]

    return np.argmax(voting_list).astype(int)

# Calcula a distancia euclidiana entre 2 pontos.
def euclidean_distance(x1: NDArray, x2: NDArray) -> float:
    return np.sqrt(np.sum((x1-x2)**2))
