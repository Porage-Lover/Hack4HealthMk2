import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_def import MultimodalNet
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Setting up hardware acceleration...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")

    print("Loading preprocessing artifacts...")
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    
    df = pd.read_csv('cleaned_psychiatric_data.csv')
    tabular_medians = df.drop(columns=['Mental_Health_Status']).median(numeric_only=True).values
    audio_zeros = np.zeros(13)

    archive_train_dir = 'archive/train'
    classes = list(le.classes_)
    emotion_mapping = {
        'happy': classes.index('Healthy'),
        'neutral': classes.index('Mild_Stress'),
        'surprise': classes.index('Mild_Stress'),
        'angry': classes.index('Moderate_Stress'),
        'disgust': classes.index('Moderate_Stress'),
        'sad': classes.index('Severe_Stress'),
        'fear': classes.index('Severe_Stress')
    }

    X_list = []
    y_list = []
    
    print("Extracting up to 21,000 images from FER dataset...")
    for emotion, label_idx in emotion_mapping.items():
        emotion_dir = os.path.join(archive_train_dir, emotion)
        if not os.path.exists(emotion_dir): continue
        images = glob.glob(os.path.join(emotion_dir, '*.jpg')) + glob.glob(os.path.join(emotion_dir, '*.png'))
        
        np.random.seed(42)
        if len(images) > 3000:
            images = np.random.choice(images, 3000, replace=False)
            
        for img_path in images:
            img = Image.open(img_path).convert('L').resize((32, 32))
            img_array = np.array(img).flatten()
            fused = np.concatenate([tabular_medians, audio_zeros, img_array])
            X_list.append(fused)
            y_list.append(label_idx)

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)

    print(f"Loaded {len(X_arr)} total samples. Preprocessing...")
    X_imp = imputer.transform(X_arr)
    X_scaled = scaler.transform(X_imp)

    X_tab = torch.tensor(X_scaled[:, :21], dtype=torch.float32)
    X_aud = torch.tensor(X_scaled[:, 21:34], dtype=torch.float32)
    X_img = torch.tensor(X_scaled[:, 34:].reshape(-1, 1, 32, 32), dtype=torch.float32)
    y_t = torch.tensor(y_arr, dtype=torch.long)

    dataset = TensorDataset(X_tab, X_aud, X_img, y_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print("Initializing Deep VGG Multimodal CNN...")
    model = MultimodalNet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for tab, aud, img, labels in loader:
            tab, aud, img, labels = tab.to(device), aud.to(device), img.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tab, aud, img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss/len(loader)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.4f} (LR: {optimizer.param_groups[0]['lr']:.5f})")

    model.to('cpu')
    torch.save(model.state_dict(), 'multimodal_cnn.pth')
    print("Saved Deep VGG weights to 'multimodal_cnn.pth'")

if __name__ == "__main__":
    main()
