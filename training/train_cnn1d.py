import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "landmarks.csv")
model_save_path = os.path.join(current_dir, "cnn_asl_model_full.pth")
encoder_save_path = os.path.join(current_dir, "label_encoder_classes.npy")

# Load data
df = pd.read_csv(data_path)
X = df.drop(columns=['label']).values
y = df['label'].values


# Normalize landmarks relative to wrist
def normalize_landmarks(landmarks):
    landmarks = landmarks.reshape(-1, 3)
    wrist = landmarks[0]
    normalized = landmarks - wrist
    scale = np.std(normalized) + 1e-6
    normalized = normalized / scale
    return normalized.flatten()

# Data augmentation
def augment_landmarks(landmarks):
    landmarks = landmarks.reshape(-1, 3)
    noise = np.random.normal(0, 0.01, landmarks.shape)
    augmented = landmarks + noise
    scale = np.random.uniform(0.9, 1.1)
    augmented = augmented * scale
    theta = np.random.uniform(-0.1, 0.1)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0, 0, 1]])
    augmented = augmented @ rotation_matrix
    return augmented.flatten()

# Normalize and augment data
X_normalized = np.array([normalize_landmarks(row) for row in X])
augment_indices = np.random.choice(len(X), size=int(0.2 * len(X)), replace=False)
X_augmented = np.array([augment_landmarks(X[i]) for i in augment_indices])
X_final = np.vstack([X_normalized, X_augmented])
y_final = np.hstack([y, y[augment_indices]])

# Reshape for PyTorch
X_final = X_final.reshape(X_final.shape[0], 1, -1).astype(np.float32)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_final)
num_classes = len(np.unique(y_encoded))

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Dataset class
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create data loaders
train_dataset = LandmarkDataset(X_train, y_train)
val_dataset = LandmarkDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# CNN model architecture
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((input_size // 4) * 32, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D(input_size=X_final.shape[2], num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(4):
    model.train()
    total_train_loss = 0
    correct_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        correct_train += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct_train / len(train_loader.dataset)
    train_losses.append(total_train_loss / len(train_loader))
    train_accs.append(train_acc)

    # Validation
    model.eval()
    total_val_loss = 0
    correct_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            correct_val += (outputs.argmax(1) == labels).sum().item()
    val_acc = correct_val / len(val_loader.dataset)
    val_losses.append(total_val_loss / len(val_loader))
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

# Save model and encoder
try:
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_final.shape[2],
        'num_classes': num_classes
    }, model_save_path)
    np.save(encoder_save_path, label_encoder.classes_)
    print(f"[INFO] Model saved to {model_save_path}")
    print(f"[INFO] Encoder saved to {encoder_save_path}")
except Exception as e:
    print(f"[ERROR] Save failed: {e}")
    exit()
