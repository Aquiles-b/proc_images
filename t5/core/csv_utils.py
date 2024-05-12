import csv 
from numpy._typing import NDArray
import numpy as np

# Escreve a lista de listas no final do arquivo csv.
def write_lists_csv(csv_name: str, hist_list: list) -> None:
    with open(csv_name, mode='a', newline=None) as hist_csv:
        writer = csv.writer(hist_csv, delimiter=';')
        writer.writerows(hist_list)

# Retorna uma lista de listas de floats vindos do csv passado.
def read_lists_csv(csv_name: str) -> list[NDArray]:
    with open(csv_name, 'r') as file:
        file_csv = csv.reader(file, delimiter=';')
        file_list = list()
        for row in file_csv:
            file_list.append(np.array([float(i) for i in row]))
        return file_list
