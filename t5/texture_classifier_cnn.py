import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

from glob import glob 


class TextureDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        self.data = sorted(glob(f'{data_dir}/*.JPG'))
        self.classes = [n.split('/')[-1][:2] for n in self.data]
        self.targets = [torch.tensor(int(i)) for i in self.classes]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]


class TextureClassifier(nn.Module):
    def __init__(self, device, lr=1e-3) -> None:
        super().__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def custom_model(self) -> None:
        self.model = nn.Sequential(
                nn.Conv2d(3, 32, (3,3)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3,3)),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3,3)),
                nn.Flatten(),
                nn.Linear(64*(3264-6)*(2448-6), 9)
                )
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_data, num_epochs=10) -> None:
        self.train()
        running_loss = 0
        for epoch in range(num_epochs):
            for images, lbl in train_data:
                self.optimizer.zero_grad()
                outs = self(images)





def main() -> None:
    transform = transforms.Compose([
        transforms.Resize((3264, 2448)), #1632, 1224
        transforms.ToTensor(),
        ])

    dataset = TextureDataset('./macroscopic0/train', transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == "__main__":
    main()
