import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast

test_dir = r'C:\Users\\test'
train_dir = r'C:\Users\\train'

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
        self.feature_extractor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor.fc = nn.Identity()  # Remove the final layer
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, num_images, c, h, w = x.size()
        x = x.view(batch_size * num_images, c, h, w)
        with autocast():
            features = self.feature_extractor(x)
        features = features.view(batch_size, num_images, -1)
        features = features.max(dim=1)[0]  # Max pooling over all images in the bag
        output = self.classifier(features)
        return output

def calculate_accuracy(outputs, labels):
    preds = torch.sigmoid(outputs) >= 0.5
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA current device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),  # Reduce size here
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = BagDataset(root_dir=train_dir, transform=transform)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset = dataset
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BagClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()
    # delete 
    import random
    num_epochs = random.randint(20, 30)
    #num_epochs = 1

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels.float().unsqueeze(1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            correct_train += calculate_accuracy(outputs, labels.float().unsqueeze(1))
            total_train += 1

        scheduler.step()
        train_accuracy = correct_train / total_train
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {train_accuracy}')

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                correct_val += calculate_accuracy(outputs, labels.float().unsqueeze(1))
                total_val += 1

        val_accuracy = correct_val / total_val
        print(f'Validation Accuracy: {val_accuracy}')
        model.train()
        if val_accuracy == 1.0 and train_accuracy == 1.0 and running_loss / len(train_loader) < 0.01:
            print("stop train")
            num_epochs = epoch +1
            break

    torch.save(model.state_dict(), f'ouput_weights\{val_accuracy}_{num_epochs}_odel_weight.pth')

    test_dataset = BagDataset(root_dir=test_dir, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    predictions = []
    image_ids = []
    with torch.no_grad():
        for images, _, file_name in tqdm(test_loader, desc='Inference', unit='batch'):
            images = images.to(device)
            with autocast():
                outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)
            image_ids.append(file_name[0])  # Use file_name[0] to get the string part of the tuple

    predictions = [1 if p >= 0.5 else 0 for p in predictions]
    image_ids = [file_name.split('.')[0] for file_name in image_ids]  # Remove .pkl extension
    submission = pd.DataFrame({'image_id': image_ids, 'y_pred': predictions})
    submission.to_csv(f'output/{val_accuracy}_{train_accuracy}_{num_epochs}_{running_loss / len(train_loader)}submission_v2.csv', index=False)

if __name__ == "__main__":
    main()
