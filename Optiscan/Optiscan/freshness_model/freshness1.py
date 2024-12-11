# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchmetrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
# Paths to datasets
TRAIN_PATH = 'dataset/Train'
TEST_PATH = 'dataset/Test'

# Load Data
def load_data(PATH):
    filenames, fruit, fresh = [], [], []
    for file in tqdm(os.listdir(PATH)):
        for img in os.listdir(os.path.join(PATH, file)):
            fresh.append(0 if file[0] == 'f' else 1)
            fruit.append(file[5:] if file[0] == 'f' else file[6:])
            filenames.append(os.path.join(PATH, file, img))
    
    df = pd.DataFrame({
        'filename': filenames,
        'fruit': fruit,
        'fresh': fresh
    })
    return df

df_train = load_data(TRAIN_PATH).sample(frac=1)
df_test = load_data(TEST_PATH).sample(frac=1)

# Cleaning and Correcting Data
df_train.drop(df_train[(df_train['fruit'] == 'capsicum') & (df_train['fruit'] == 'bittergourd')].index, inplace=True)
df_test['fruit'] = df_test['fruit'].map(lambda x: 'tomato' if x == 'tamto' else x)
df_test['fruit'] = df_test['fruit'].map(lambda x: 'potato' if x == 'patato' else x)
df = pd.concat([df_train, df_test], axis=0)

# Create balanced dataset
counts = df['fruit'].value_counts()
df_new = pd.DataFrame(columns=['filename', 'fruit', 'fresh'])
for key, value in counts.items():
    df_temp = df[df['fruit'] == key].sample(n=1500) if value > 1500 else df[df['fruit'] == key]
    df_new = pd.concat([df_new, df_temp], axis=0)

# Label Encoding
le = LabelEncoder()
df_new['fruit_label'] = le.fit_transform(df_new['fruit'])

# Split into training and validation sets
df_train, df_val = train_test_split(df_new, test_size=0.15, stratify=df_new['fruit_label'])

# Define Helper Functions
def load_image(path):
    return plt.imread(path)

def image_transform(img, p=0.5, training=True):
    # Ensure the image is writable by making a copy
    img = np.copy(img)

    # Proceed with the rest of the transformation
    transforms_list = [transforms.ToTensor(), transforms.Resize((224, 224))]
    if training:
        transforms_list += [transforms.RandomHorizontalFlip(p=p),
                            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                            transforms.RandomAdjustSharpness(3, p=p)]
    transforms_list.append(transforms.Normalize(mean=0, std=1))
    
    img = transforms.Compose(transforms_list)(img)
    return img

# Dataset Class
class FruitDataset(Dataset):
    def __init__(self, df, training):
        self.df = df
        self.training = training
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img = plt.imread(self.df.iloc[idx]['filename'])[:, :, :3]
        fresh = torch.tensor(self.df.iloc[idx]['fresh'])
        fruit = torch.tensor(self.df.iloc[idx]['fruit_label'])
        img = image_transform(img, training=self.training)
        return img, fruit, fresh

# Dataloaders
BATCH_SIZE = 64
train_dataset = FruitDataset(df_train, training=True)
val_dataset = FruitDataset(df_val, training=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Define the Model
# Define the Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.7
        # Updated the deprecated 'pretrained' argument
        self.base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all but the last few layers
        for param in list(self.base.parameters())[:-15]:  # Adjust the number of layers to freeze if necessary
            param.requires_grad = False
        
        self.base.fc = nn.Sequential()  # Ensure no inplace modification

        # Customize classifier blocks
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(128, 9)
        )

        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

        # Optimizers
        self.optimizer1 = torch.optim.Adam([{'params': self.base.parameters(), 'lr': 1e-5},
                                            {'params': self.block1.parameters(), 'lr': 3e-4}])
        self.optimizer2 = torch.optim.Adam(self.block2.parameters(), lr=3e-4)
        self.optimizer3 = torch.optim.Adam(self.block3.parameters(), lr=3e-4)

        # Loss function and metrics
        self.loss_fxn = nn.CrossEntropyLoss()
        self.fruit_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=9).to(device)
        self.fresh_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)

        # History tracking
        self.history = {'train_loss': [], 'val_loss': [], 
                        'train_acc_fruit': [], 'train_acc_fresh': [],
                        'val_acc_fruit': [], 'val_acc_fresh': []}

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        y1 = self.block2(x)
        y2 = self.block3(x)
        return y1, y2

    def train_step(self, x, y1, y2):
    # Forward pass
        pred1, pred2 = self.forward(x)

    # Loss calculations
        l1, l2 = self.loss_fxn(pred1, y1), self.loss_fxn(pred2, y2)
    
    # Combined loss
        loss = self.alpha * l1 + (1 - self.alpha) * l2
    
    # Zero gradients for all optimizers
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
    
    # Backward pass (once for the combined loss)
        loss.backward()
    
    # Optimizer steps
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer3.step()

    # Accuracy calculations
        fruit_acc = self.fruit_accuracy(torch.argmax(pred1, axis=1), y1)
        fresh_acc = self.fresh_accuracy(torch.argmax(pred2, axis=1), y2)

        return loss, fruit_acc, fresh_acc


    def val_step(self, x, y1, y2):
        with torch.no_grad():
            pred1, pred2 = self.forward(x)
            loss = self.alpha * self.loss_fxn(pred1, y1) + (1 - self.alpha) * self.loss_fxn(pred2, y2)
            fruit_acc = self.fruit_accuracy(torch.argmax(pred1, axis=1), y1)
            fresh_acc = self.fresh_accuracy(torch.argmax(pred2, axis=1), y2)
            return loss, fruit_acc, fresh_acc
        
    def update_history(self, train_loss, train_fruit, train_fresh, val_loss, val_fruit, val_fresh):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc_fresh'].append(train_fresh)
        self.history['train_acc_fruit'].append(train_fruit)
        self.history['val_acc_fresh'].append(val_fresh)
        self.history['val_acc_fruit'].append(val_fruit)

    def train_model(self, epochs):
        for epoch in tqdm(range(epochs)):
            train_loss, train_fruit, train_fresh = 0, 0, 0
            val_loss, val_fruit, val_fresh = 0, 0, 0

            train_batches, val_batches = 0, 0  # Count batches

            # Training Loop
            self.train()
            for X, y1, y2 in train_loader:
                X, y1, y2 = [v.to(device) for v in (X, y1, y2)]
                loss, fruit_acc, fresh_acc = self.train_step(X, y1, y2)
                train_loss += loss.item()
                train_fruit += fruit_acc.item()
                train_fresh += fresh_acc.item()
                train_batches += 1  # Increment batch counter

            # Validation Loop
            self.eval()
            for X, y1, y2 in val_loader:
                X, y1, y2 = [v.to(device) for v in (X, y1, y2)]
                loss, fruit_acc, fresh_acc = self.val_step(X, y1, y2)
                val_loss += loss.item()  
                val_fruit += fruit_acc.item()
                val_fresh += fresh_acc.item()
                val_batches += 1  # Increment batch counter

            # Normalize accumulated values by number of batches
            train_loss /= train_batches
            train_fruit /= train_batches
            train_fresh /= train_batches
            val_loss /= val_batches
            val_fruit /= val_batches
            val_fresh /= val_batches

            self.update_history(train_loss, train_fruit, train_fresh, val_loss, val_fruit, val_fresh)

            # Print epoch results
            print(f"[Epoch {epoch+1}] Train: [Loss: {train_loss:.3f}] [Fruit Accuracy: {train_fruit*100:.2f}%] [Fresh Accuracy: {train_fresh*100:.2f}%]")
            print(f"Validation: [Loss: {val_loss:.3f}] [Fruit Accuracy: {val_fruit*100:.2f}%] [Fresh Accuracy: {val_fresh*100:.2f}%]")



model = Model().to(device)
model.train_model(epochs=6)
torch.save(model.state_dict(), "fruit_fresh_model.pth")