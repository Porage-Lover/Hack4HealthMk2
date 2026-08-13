import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import torch
import torch.nn as nn
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
    print("Loading PyTorch CNN model and artifacts...")
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    
    model = MultimodalNet()
    model.load_state_dict(torch.load('multimodal_cnn.pth', weights_only=True))
    model.eval()

    df = pd.read_csv('cleaned_psychiatric_data.csv')
    tabular_medians = df.drop(columns=['Mental_Health_Status']).median(numeric_only=True).values
    audio_zeros = np.zeros(13)

    archive_test_dir = 'archive/test'
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    print("\nEvaluating PyTorch CNN against FER-2013 external images...")
    print(f"{'True Emotion':<12} | {'Predicted Mental Health Status Distribution'}")
    print("-" * 85)

    for emotion in emotions:
        emotion_dir = os.path.join(archive_test_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        images = glob.glob(os.path.join(emotion_dir, '*.jpg')) + glob.glob(os.path.join(emotion_dir, '*.png'))
        images = images[:200]
        
        if len(images) == 0:
            continue

        predictions = []
        for img_path in images:
            img = Image.open(img_path).convert('L').resize((32, 32))
            img_array = np.array(img).flatten()
            
            fused = np.concatenate([tabular_medians, audio_zeros, img_array]).reshape(1, -1)
            fused_imp = imputer.transform(fused)
            fused_scaled = scaler.transform(fused_imp)
            
            tab_t = torch.tensor(fused_scaled[:, :21], dtype=torch.float32)
            aud_t = torch.tensor(fused_scaled[:, 21:34], dtype=torch.float32)
            img_t = torch.tensor(fused_scaled[:, 34:].reshape(-1, 1, 32, 32), dtype=torch.float32)
            
            with torch.no_grad():
                out = model(tab_t, aud_t, img_t)
                _, pred_idx = torch.max(out, 1)
                
            pred_label = le.inverse_transform(pred_idx.numpy())[0]
            predictions.append(pred_label)
            
        unique, counts = np.unique(predictions, return_counts=True)
        total = len(predictions)
        sorted_indices = np.argsort(-counts)
        dist_str = ", ".join([f"{unique[i]}: {counts[i]/total*100:.1f}%" for i in sorted_indices])
        print(f"{emotion.capitalize():<12} | {dist_str}")

if __name__ == "__main__":
    main()
