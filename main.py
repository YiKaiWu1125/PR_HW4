import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import pandas as pd
from tqdm import tqdm

class BagDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.is_test = is_test
        self._load_data()

    def _load_data(self):
        if self.is_test:
            for file_name in os.listdir(self.root_dir):
                file_path = os.path.join(self.root_dir, file_name)
                with open(file_path, 'rb') as f:
                    images = pickle.load(f)
                    for img in images:
                        self.data.append(img)
                        self.labels.append(-1)  # Dummy label for test data
        else:
            for label, class_dir in enumerate(['class_0', 'class_1']):
                class_dir_path = os.path.join(self.root_dir, class_dir)
                for file_name in os.listdir(class_dir_path):
                    file_path = os.path.join(class_dir_path, file_name)
                    with open(file_path, 'rb') as f:
                        images = pickle.load(f)
                        for img in images:
                            self.data.append(img)
                            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class BagClassifier(nn.Module):
    def __init__(self):
        super(BagClassifier, self).__init__()
        self.feature_extractor = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor.fc = nn.Identity()  # Remove the final layer
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA current device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = BagDataset(root_dir=r'C:\Users\YK\Desktop\113HW4\released\train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = BagDataset(root_dir=r'C:\Users\YK\Desktop\113HW4\released\test', transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BagClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

    torch.save(model.state_dict(), 'model_weights.pth')

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc='Inference', unit='batch'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)

    submission = pd.DataFrame({'image_id': [f'image_{i}' for i in range(len(predictions))], 'Label': predictions})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
