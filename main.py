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
import numpy as np

class BagDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.is_test = is_test
        self.file_names = []
        self._load_data()

    def _load_data(self):
        if self.is_test:
            for file_name in os.listdir(self.root_dir):
                file_path = os.path.join(self.root_dir, file_name)
                with open(file_path, 'rb') as f:
                    images = pickle.load(f)
                    self.data.append(images)
                    self.file_names.append(file_name)
                    self.labels.append(-1)  # Dummy label for test data
        else:
            for label, class_dir in enumerate(['class_0', 'class_1']):
                class_dir_path = os.path.join(self.root_dir, class_dir)
                for file_name in os.listdir(class_dir_path):
                    file_path = os.path.join(class_dir_path, file_name)
                    with open(file_path, 'rb') as f:
                        images = pickle.load(f)
                        self.data.append(images)
                        self.file_names.append(file_name)
                        self.labels.append(label)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        images = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            images = [self.transform(img) for img in images]

        images = torch.stack(images)
        return images, label, self.file_names[idx]

class BagClassifier(nn.Module):
    def __init__(self):
        super(BagClassifier, self).__init__()
        self.feature_extractor = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor.fc = nn.Identity()  # Remove the final layer
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        batch_size, num_images, c, h, w = x.size()
        x = x.view(batch_size * num_images, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_images, -1)
        features = features.max(dim=1)[0]  # Max pooling over all images in the bag
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
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = BagDataset(root_dir=r'C:\Users\YK\Desktop\113HW4\released\train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Batch size is 1 since each bag is a batch

    test_dataset = BagDataset(root_dir=r'C:\Users\YK\Desktop\113HW4\released\test', transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BagClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    num_epochs = 1

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

    torch.save(model.state_dict(), 'model_weights.pth')

    model.eval()
    predictions = []
    image_ids = []
    with torch.no_grad():
        for images, _, file_name in tqdm(test_loader, desc='Inference', unit='batch'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)
            image_ids.append(file_name)

    predictions = [1 if p >= 0.5 else 0 for p in predictions]
    image_ids = [file_name.split('.')[0] for file_name in image_ids]  # Remove .pkl extension
    submission = pd.DataFrame({'image_id': image_ids, 'y_pred': predictions})
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
