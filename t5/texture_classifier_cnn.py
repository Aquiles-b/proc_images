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
            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            self.eval()
            running_loss = 0
            with torch.no_grad():
                for images, lbls in val_loader:
                    images, lbls = images.to(self.device), lbls.to(self.device)
                    outs = self(images)
                    loss = criterion(outs, lbls)
                    running_loss += loss.item() * images.size(0)
                val_loss = running_loss / len(val_loader.dataset)
                val_losses.append(val_loss)

            # Epoch stats
            print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    def predict(self, image) -> dict:
        self.eval()
        if isinstance(image, str):
            image = Image.open(image)
            image = transforms.ToTensor()(image)
        elif not torch.is_tensor(image):
            image = transforms.ToTensor()(image)
        
        image = image.to(self.device)
        with torch.no_grad():
            out = self(image)
            probs = nn.functional.softmax(out, dim=1)

        idx = torch.argmax(probs)
        pred = {'class': idx.item(), 'prob': probs.tolist()[idx]}

        return pred

def main() -> None:
    transform = transforms.Compose([
        transforms.Resize((3264, 2448)), #1632, 1224
        transforms.ToTensor(),
        ])

    # Carrega os conjuntos de dados para treino
    train_data = TextureDataset('./macroscopic0/train', transform)
    val_data = TextureDataset('./macroscopic0/val', transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

    model = TextureClassifier('cpu')
    model.create_custom_model((2448, 3264))
    # model.train_model(train_loader, val_loader, 1e-3)
    pred = model.predict('./macroscopic0/val/0101.JPG')
    print(pred)

if __name__ == "__main__":
    main()
