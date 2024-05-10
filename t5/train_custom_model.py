from texture_classifier_cnn import TextureClassifier, TextureDataset
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import torch
import sys
import os


def calc_input_mlp(image_dim: tuple[int, int], cnn: nn.Sequential) -> int:
	x = torch.randn(1, 3, *image_dim)
	y = cnn(x)
	return y.view(1, -1).shape[1]

def create_custom_model(image_dim: tuple[int, int], 
				num_classes: int = 9) -> tuple[nn.Sequential, nn.Sequential, nn.Sequential]:

	cnn1 = nn.Sequential(
		nn.Conv2d(3, 32, kernel_size=(3,3)),
		nn.ReLU(),
		nn.BatchNorm2d(32),
		nn.AvgPool2d(kernel_size=(2,2), stride=2),

		nn.Conv2d(32, 64, kernel_size=(3,3)),
		nn.ReLU(),
		nn.BatchNorm2d(64),
		nn.AvgPool2d(kernel_size=(4,4), stride=4),

		nn.Conv2d(64, 64, kernel_size=(3,3)),
		nn.ReLU(),
		nn.BatchNorm2d(64),
		nn.AvgPool2d(kernel_size=(2,2), stride=2)
		)

	cnn2 = nn.Sequential(
		nn.Conv2d(3, 32, kernel_size=(3,3)),
		nn.ReLU(),
		nn.Conv2d(32, 64, kernel_size=(3,3)),
		nn.ReLU(),
		nn.BatchNorm2d(64),
		nn.AvgPool2d(kernel_size=(2,2), stride=2),

		nn.Conv2d(64, 64, kernel_size=(3,3)),
		nn.ReLU(),
		nn.BatchNorm2d(64),
		nn.AvgPool2d(kernel_size=(2,2), stride=2),
		)

	fc1_input = calc_input_mlp(image_dim, cnn1)
	fc1_input += calc_input_mlp(image_dim, cnn2)

	classifier = nn.Sequential(
		nn.Linear(fc1_input, 512),
		nn.ReLU(),
		nn.Linear(512, 180),
		nn.ReLU(),
		nn.Linear(180, num_classes)
		)

	return cnn1, cnn2, classifier

def main() -> None:
	np.random.seed(2024)
	torch.manual_seed(2024)

	image_dim_original = (2448, 3264)
	fator = 4
	image_dim = (int(image_dim_original[0]/fator), int(image_dim_original[1]/fator))
	transform = transforms.Compose([
		transforms.RandomRotation(360),
		transforms.RandomCrop((768, 768)),
		transforms.Resize((384, 384)),
		transforms.RandomHorizontalFlip(0.5),
		transforms.RandomVerticalFlip(0.5),
		transforms.ToTensor(),
		])

	# Carrega os conjuntos de dados para treino
	train_data = TextureDataset('./macroscopic0/train', transform)
	val_data = TextureDataset('./macroscopic0/val', transform)

	batch_size = 16

	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

	cnn1, cnn2, clf = create_custom_model((384, 384), 9)

	model = TextureClassifier(sys.argv[1])
	model.custom_model(cnn1, cnn2, clf)
	if os.path.exists("texture_classifier.pt"):
		model.load_model("texture_classifier.pt")

	start = time.time()
	print('Começando treino:')
	model.train_model(train_loader, val_loader, 0.0001, 30)
	print(f'{(time.time() - start)/60} minutos')


if __name__ == "__main__":
	main()
