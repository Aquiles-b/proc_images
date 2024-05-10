from core import TextureClassifier, TextureDataset
from generate_data import create_custom_model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

model = TextureClassifier("cuda")
model.custom_model(*create_custom_model((384, 384), 9))
model.load_model("texture_classifier384-384.pt")
transform = transforms.Compose([
		transforms.CenterCrop((768, 768)),
		transforms.Resize((384, 384)),
		transforms.ToTensor(),
	])
dataset = TextureDataset("macroscopic0/test", transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
correct, miss = 0, 0

confusion_matrix = [[0] * 9 for _ in range(9)]

for img, lbl in test_loader:
	out = model.predict(img)
	confusion_matrix[lbl][out["class"]] += 1
	if lbl == out["class"]: correct += 1
	else: miss += 1
print(correct, miss)
for i, line in enumerate(confusion_matrix):
	total = sum(line)
	print(line, f"{line[i]/total:.2f}")
