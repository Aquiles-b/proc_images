import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from glob import glob 


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
    def __init__(self, device) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.to(self.device)

    def _calc_out_dim_model(self, x: int) -> int:
        for _ in range(3):
            x -= 2
            x = x // 4
        x -= 2
        x = x // 2
        return x

    def create_custom_model(self, image_dimension: tuple[int, int]) -> None:
        h, w = image_dimension
        self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, (3,3)),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(4,4), stride=4),
                nn.Conv2d(64, 32, (3,3)),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(4,4), stride=4),
                nn.Conv2d(32, 32, (3,3)),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(4,4), stride=4),
                nn.Conv2d(32, 16, (3,3)),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(2,2), stride=2),
                nn.Flatten()
                )
        fc1_tam = self._calc_out_dim_model(w) * self._calc_out_dim_model(h) * 16
        self.classifier = nn.Sequential(
                nn.Linear(fc1_tam, 180),
                nn.Linear(180, 45),
                nn.Linear(45, 9)
                )

        self.model = nn.Sequential(self.cnn, self.classifier)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                    lr: float, num_epochs: int = 10) -> None:

        optimizer = optim.Adam(self.model.parameters(), lr)
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
                optimizer.zero_grad()
                outs = self(images)
                loss = criterion(outs, lbls)
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
        
        torch.save(self.state_dict(), './texture_classifier.pt')

    def predict(self, image) -> dict:
        self.eval()
        if isinstance(image, str):
            image = Image.open(image)
            image = transforms.ToTensor()(image)
        elif not torch.is_tensor(image):
            image = transforms.ToTensor()(image)

        # Adiciona uma dimensão para o tamanho do lote (batch)
        image = image.unsqueeze(0)
        
        image = image.to(self.device)
        with torch.no_grad():
            out = self(image)
            probs = nn.functional.softmax(out, dim=1)

        idx = torch.argmax(probs).item()
        pred = {'class': idx, 'prob': probs.tolist()[0][idx]}

        return pred

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
