from numpy._typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image

from glob import glob 


class LBPDataset(Dataset):
    def __init__(self, hists: list[NDArray]):
        self.hists = [hist[:-1] for hist in hists]
        self.targets = [int(hist[-1]) for hist in hists]

    def __len__(self):
        return len(self.hists)

    def __getitem__(self, idx):
        return torch.tensor(self.hists[idx], dtype=torch.float32), torch.tensor(self.targets[idx])


class TextureDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        self.data = sorted(glob(f'{data_dir}/*.JPG'))
        self.classes = [n.split('/')[-1][:2] for n in self.data]
        self.targets = [torch.tensor(int(i)-1) for i in self.classes]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]


class TextureClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def custom_model(self, cnn1: nn.Sequential, cnn2: nn.Sequential, classifier: nn.Sequential) -> None:
        self.cnn1 = cnn1
        self.cnn2 = cnn2
        self.classifier = classifier
        self.cnn1.to(self.device)
        self.cnn2.to(self.device)
        self.classifier.to(self.device)

        self.forward = self.custom_forward

    def VGG16(self, num_classes: int, freeze: bool = True) -> None:
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.cnn1 = vgg16.features
        self.cnn2 = None
        if freeze:
            for param in self.cnn1.parameters():
                param.requires_grad = False

        tam_fc1 = vgg16.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(tam_fc1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.cnn1.to(self.device)
        self.classifier.to(self.device)

    def create_LBP_clf(self, num_inputs: int, num_classes: int) -> None:
        self.classifier = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.classifier.to(self.device)
        self.forward = self.forward_simple_classifier

    def forward_simple_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        x = self.cnn1(x)
        tam_x1 = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, tam_x1)
        x = self.classifier(x)

        return x

    def custom_forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)

        tam_x1 = x1.size(1) * x1.size(2) * x1.size(3)
        tam_x2 = x2.size(1) * x2.size(2) * x2.size(3)

        x = torch.cat((x1.view(-1, tam_x1), x2.view(-1, tam_x2)), dim=1)

        x = self.classifier(x)

        return x

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                    lr: float, num_epochs: int = 10, path_to_save: str = '.') -> None:

        optimizer = optim.Adam(self.parameters(), lr)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []
        tam_train_loader = len(train_loader.dataset)
        tam_val_loader = len(val_loader.dataset)

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0
            for images, lbls in train_loader:
                images, lbls = images.to(self.device), lbls.to(self.device)
                outs = self(images)
                loss = criterion(outs, lbls)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            train_loss = running_loss / tam_train_loader
            train_losses.append(train_loss)

            self.eval()
            running_loss = 0
            with torch.no_grad():
                for images, lbls in val_loader:
                    images, lbls = images.to(self.device), lbls.to(self.device)
                    outs = self(images)
                    loss = criterion(outs, lbls)
                    running_loss += loss.item() * images.size(0)
                val_loss = running_loss / tam_val_loader
                val_losses.append(val_loss)

            # Epoch stats
            print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

        torch.save(self.state_dict(), path_to_save)

    def predict(self, x) -> dict:
        self.eval()

        x = x.to(self.device)
        with torch.no_grad():
            out = self(x)
            probs = nn.functional.softmax(out, dim=1)

        idx = torch.argmax(probs).item()
        pred = {'class': idx, 'prob': probs.tolist()[0][idx]}

        return pred

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
