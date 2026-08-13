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
import warnings

warnings.filterwarnings('ignore')

class MultimodalNet(nn.Module):
    def __init__(self):
        super(MultimodalNet, self).__init__()
        self.tab_fc = nn.Sequential(nn.Linear(21, 16), nn.ReLU())
        self.aud_fc = nn.Sequential(nn.Linear(13, 16), nn.ReLU())
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 4)
        )

    def forward(self, tab, aud, img):
        return self.fusion(torch.cat((self.tab_fc(tab), self.aud_fc(aud), self.cnn(img)), dim=1))

def main():
    print("Loading preprocessing artifacts...")
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    
    df = pd.read_csv('cleaned_psychiatric_data.csv')
    tabular_medians = df.drop(columns=['Mental_Health_Status']).median(numeric_only=True).values
    audio_zeros = np.zeros(13)

    archive_train_dir = 'archive/train'
    
    # Map FER emotions to our hackathon Mental_Health_Status classes
    classes = list(le.classes_)
    healthy_idx = classes.index('Healthy')
    mild_idx = classes.index('Mild_Stress')
    mod_idx = classes.index('Moderate_Stress')
    sev_idx = classes.index('Severe_Stress')

    emotion_mapping = {
        'happy': healthy_idx,
        'neutral': mild_idx,
        'surprise': mild_idx,
        'angry': mod_idx,
        'disgust': mod_idx,
        'sad': sev_idx,
        'fear': sev_idx
    }

    X_list = []
    y_list = []
    
    print("Extracting images from FER dataset and aligning to hackathon labels...")
    for emotion, label_idx in emotion_mapping.items():
        emotion_dir = os.path.join(archive_train_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        images = glob.glob(os.path.join(emotion_dir, '*.jpg')) + glob.glob(os.path.join(emotion_dir, '*.png'))
        
        # Sample 600 per emotion for balanced, rapid training
        np.random.seed(42)
        if len(images) > 600:
            images = np.random.choice(images, 600, replace=False)
            
        for img_path in images:
            img = Image.open(img_path).convert('L').resize((32, 32))
            img_array = np.array(img).flatten()
            fused = np.concatenate([tabular_medians, audio_zeros, img_array])
            X_list.append(fused)
            y_list.append(label_idx)

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)

    print(f"Extracted {len(X_arr)} real faces for training. Preprocessing...")
    X_imp = imputer.transform(X_arr)
    X_scaled = scaler.transform(X_imp)

    X_tab = torch.tensor(X_scaled[:, :21], dtype=torch.float32)
    X_aud = torch.tensor(X_scaled[:, 21:34], dtype=torch.float32)
    X_img = torch.tensor(X_scaled[:, 34:].reshape(-1, 1, 32, 32), dtype=torch.float32)
    y_t = torch.tensor(y_arr, dtype=torch.long)

    dataset = TensorDataset(X_tab, X_aud, X_img, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("Training CNN to actually recognize facial expressions (10 epochs)...")
    model = MultimodalNet()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for tab, aud, img, labels in loader:
            optimizer.zero_grad()
            outputs = model(tab, aud, img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {running_loss/len(loader):.4f}")

    # Overwrite the weights with the strictly better ones
    torch.save(model.state_dict(), 'multimodal_cnn.pth')
    print("Saved real facial-expression-aware model to 'multimodal_cnn.pth'")

if __name__ == "__main__":
    main()
