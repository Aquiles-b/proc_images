#!/bin/python3

from math import ceil
import urllib.request
import zipfile
import os
from glob import glob
import random


# Download and extract the dataset.
def download_and_extract() -> None:
    link = 'https://zenodo.org/records/10219797/files/macroscopic0.zip?download=1'
    file_name = 'macroscopic0.zip'

    print(f'Downloading {file_name}...')
    urllib.request.urlretrieve(link, file_name)

    print('Extracting the zip file...')
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall('macroscopic0')

    os.remove(file_name)
    print('Completed!')

# Separate the images into different lists based on first two 
# characters of the image name.
def organize_by_class(images) -> dict:
    imgs_by_classes = {}
    for image in images:
        class_name = image.split('/')[-1][:2]
        if class_name not in imgs_by_classes:
            imgs_by_classes[class_name] = []
        imgs_by_classes[class_name].append(image)

    return imgs_by_classes

# Copy n random images to a directory randomly and remove them from the list.
# THIS FUNCTION CONSUMES THE LIST!
def n_random_images_to_dir(imgs, num_imgs, dir_name) -> None:
    os.makedirs(dir_name, exist_ok=True)
    for _ in range(num_imgs):
        img = random.choice(imgs)
        os.rename(img, f'{dir_name}/{img.split("/")[-1]}')
        imgs.remove(img)

# Separate the data into train, val, and test, organized in directories.
def divide_train_val_test() -> None:
    images = sorted(glob('macroscopic0/*.JPG'))
    if len(images) == 0:
        return
    imgs_by_classes = organize_by_class(images)

    for label in imgs_by_classes:
        imgs = imgs_by_classes[label]
        num_imgs = len(imgs)
        num_test = ceil(num_imgs * 0.5)
        num_train = ceil(num_imgs * 0.325)

        n_random_images_to_dir(imgs, num_train, 'macroscopic0/train')
        n_random_images_to_dir(imgs, num_test, 'macroscopic0/test')
        n_random_images_to_dir(imgs, len(imgs), 'macroscopic0/val')

def main() -> None:
    # Set the seed for reproducibility.
    random.seed(2024)

    if not os.path.exists('macroscopic0'):
        download_and_extract()

    divide_train_val_test()


if __name__ == '__main__':
    main()
