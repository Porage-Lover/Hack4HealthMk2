import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

from model_def import MultimodalNet

def main():
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

    y_true = []
    y_pred = []

    print("Running full accuracy benchmark on unseen FER-2013 test set...")
    
    for emotion, label_idx in emotion_mapping.items():
        emotion_dir = os.path.join(archive_test_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        images = glob.glob(os.path.join(emotion_dir, '*.jpg')) + glob.glob(os.path.join(emotion_dir, '*.png'))
        # Evaluate up to 500 images per class
        images = images[:500]
        
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
                
            y_true.append(label_idx)
            y_pred.append(pred_idx.item())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n" + "="*40)
    print(" EXTERNAL BENCHMARK RESULTS")
    print("="*40)
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Macro Precision  : {prec * 100:.2f}%")
    print(f"Macro Recall     : {rec * 100:.2f}%")
    print(f"Macro F1-Score   : {f1 * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
