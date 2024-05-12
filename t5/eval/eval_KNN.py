import os
import sys

from numpy._typing import NDArray
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import KNN_decision

def evaluate_KNN(train_hists: list[NDArray], test_hists: list[NDArray],
                 num_classes : int = 9, knn: int = 1) -> list[list[int]]:
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    for hist in test_hists:
        decision = KNN_decision(hist, train_hists, knn=knn, num_classes=num_classes)
        lbl = int(hist[-1])
        confusion_matrix[lbl][decision] += 1

    return confusion_matrix


