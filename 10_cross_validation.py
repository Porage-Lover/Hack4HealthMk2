import os
import glob
import numpy as np
import pandas as pd
import librosa
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Loading fine-tuned ResNet-18 Feature Extractor...")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load('fer_resnet18_best.pth', map_location=device, weights_only=True))
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    audio_dir = 'drive-download-20260813T032549Z-1-001/Audios'
    image_dir = 'drive-download-20260813T032549Z-1-001/Extracted_images'
    wav_files = sorted(glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True))
    png_files = sorted(glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True))
    
    df = pd.read_csv('cleaned_psychiatric_data.csv')
    
    min_len = min(len(wav_files), len(png_files), len(df))
    wav_files = wav_files[:min_len]
    png_files = png_files[:min_len]
    df = df.iloc[:min_len]
    
    print(f"Extracting Multimodal Features for {min_len} samples... (This takes a moment)")
    X_features = []
    for i in range(min_len):
        img = Image.open(png_files[i]).convert('L')
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vis_emb = model(img_t).cpu().numpy().flatten()
            
        y_audio, sr = librosa.load(wav_files[i], sr=16000, duration=2.0)
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
        aud_emb = np.mean(mfcc, axis=1)
        
        tab_row = df.drop(columns=['Mental_Health_Status']).iloc[i].values
        X_features.append(np.concatenate([vis_emb, aud_emb, tab_row]))
        
    X = np.array(X_features)
    le = LabelEncoder()
    y = le.fit_transform(df['Mental_Health_Status'])
    
    print("Executing 5-Fold Stratified Cross Validation...")
    # We must use imblearn's Pipeline to ensure SMOTE is only applied to the training folds!
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=1, tree_method='hist'))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    
    print("\n" + "="*40)
    print(" 5-FOLD CROSS-VALIDATION RESULTS ")
    print("="*40)
    for i, score in enumerate(scores):
        print(f"Fold {i+1} Accuracy : {score * 100:.2f}%")
    print("-" * 40)
    print(f"Average Accuracy : {np.mean(scores) * 100:.2f}%")
    print(f"Std Deviation    : ±{np.std(scores) * 100:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()
