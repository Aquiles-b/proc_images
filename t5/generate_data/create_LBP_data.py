import __basic_import__
from core import create_LBP_csv_hists
import os


def create_LBP_data_sets() -> None:
    CURRENT_DIR = os.path.dirname(__file__)
    macroscopic0_dir = f'{CURRENT_DIR}/../macroscopic0'
    data_dir = f'{CURRENT_DIR}/../data'

    image_dim = (3264, 2448)
    fator = 4
    image_dim = (image_dim[0] // fator, image_dim[1] // fator)

    create_LBP_csv_hists(f'{macroscopic0_dir}/train', 
                         f'{data_dir}/train_LBP.csv', image_dim)

    create_LBP_csv_hists(f'{macroscopic0_dir}/test',
                         f'{data_dir}/test_LBP.csv', image_dim)

    create_LBP_csv_hists(f'{macroscopic0_dir}/val',
                         f'{data_dir}/val_LBP.csv', image_dim)

if __name__ == '__main__':
    create_LBP_data_sets()
