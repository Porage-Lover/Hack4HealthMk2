import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Load the data
    print("Loading preprocessed scaled data...")
    X_train = np.load('X_train_mega.npy')
    y_train = np.load('y_train_mega.npy')
    X_test = np.load('X_test_mega.npy')
    y_test = np.load('y_test_mega.npy')

    # 2. Slice the features back into their modalities
    # Tabular: 0 to 20 (21 features)
    # Audio: 21 to 33 (13 features)
    # Image: 34 to 1057 (1024 features)
    
    X_train_tab = torch.tensor(X_train[:, :21], dtype=torch.float32)
    X_train_aud = torch.tensor(X_train[:, 21:34], dtype=torch.float32)
    # Reshape image to (N, Channels, H, W) -> (N, 1, 32, 32)
    X_train_img = torch.tensor(X_train[:, 34:].reshape(-1, 1, 32, 32), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_test_tab = torch.tensor(X_test[:, :21], dtype=torch.float32)
    X_test_aud = torch.tensor(X_test[:, 21:34], dtype=torch.float32)
    X_test_img = torch.tensor(X_test[:, 34:].reshape(-1, 1, 32, 32), dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 3. Create DataLoaders
    train_ds = TensorDataset(X_train_tab, X_train_aud, X_train_img, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # 4. Define the True Multimodal Network
    class MultimodalNet(nn.Module):
        def __init__(self):
            super(MultimodalNet, self).__init__()
            
            # Tabular branch
            self.tab_fc = nn.Sequential(
                nn.Linear(21, 16),
                nn.ReLU()
            )
            
            # Audio branch
            self.aud_fc = nn.Sequential(
                nn.Linear(13, 16),
                nn.ReLU()
            )
            
            # Image branch (CNN to learn actual spatial features from the pixels)
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), # Reduces 32x32 to 16x16
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), # Reduces 16x16 to 8x8
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 64),
                nn.ReLU()
            )
            
            # Fusion layer
            # 16 (tab) + 16 (aud) + 64 (img) = 96
            self.fusion = nn.Sequential(
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(0.4), # Prevent overfitting to the images
                nn.Linear(64, 4) # 4 output classes for Mental Health Status
            )

        def forward(self, tab, aud, img):
            tab_out = self.tab_fc(tab)
            aud_out = self.aud_fc(aud)
            img_out = self.cnn(img)
            
            # Concatenate the embeddings from all 3 modalities
            merged = torch.cat((tab_out, aud_out, img_out), dim=1)
            out = self.fusion(merged)
            return out

    model = MultimodalNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Train the model
    print("Training True Multimodal CNN (PyTorch)...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for tab, aud, img, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(tab, aud, img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

    # 6. Evaluate
    print("\nEvaluating on Test Set...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tab, X_test_aud, X_test_img)
        _, preds = torch.max(outputs, 1)

    acc = accuracy_score(y_test, preds.numpy())
    print(f"\nFinal CNN Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds.numpy()))

    # 7. Save model
    torch.save(model.state_dict(), 'multimodal_cnn.pth')
    print("\nSaved PyTorch model architecture weights to 'multimodal_cnn.pth'")
    
if __name__ == "__main__":
    main()
