"""
Multimodal Media Processor and Preprocessing Pipeline
===================================================
1. Extracts raw features from Audios and Images.
2. Aligns them alphanumerically and concatenates with the tabular CSV.
3. Applies Preprocessing (Imputation, Standard Scaling, Train-Test splits) AFTER Mega-DataFrame is fully constructed.
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
import joblib

# Suppress librosa load warnings for cleaner output
warnings.filterwarnings('ignore')

def extract_audio_features(wav_path):
    try:
        # Load audio (downsampled to 16kHz for speed, max 2 seconds)
        y, sr = librosa.load(wav_path, sr=16000, duration=2.0)
        # Extract 13 MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Average over the time axis to get a 1D feature vector
        return np.mean(mfcc, axis=1)
    except Exception as e:
        # Fallback to zeros in case of unreadable file
        return np.zeros(13)

def extract_image_features(png_path):
    try:
        # Open image and convert to grayscale
        img = Image.open(png_path).convert('L')
        # Resize to 32x32 to manage feature dimensionality (1024 features)
        img = img.resize((32, 32))
        # Flatten into a 1D array
        return np.array(img).flatten()
    except Exception as e:
        # Fallback to zeros
        return np.zeros(32 * 32)

def main():
    print("--- Phase 1 & 2: Raw Media Extraction & Feature Generation ---")
    audio_dir = 'drive-download-20260813T032549Z-1-001/Audios'
    image_dir = 'drive-download-20260813T032549Z-1-001/Extracted_images'
    csv_path = 'cleaned_psychiatric_data.csv'

    # 1. Alphanumerically sort files
    wav_files = sorted(glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True))
    png_files = sorted(glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True))

    print(f"Found {len(wav_files)} audio files and {len(png_files)} image files.")

    # Load tabular data
    df = pd.read_csv(csv_path)
    print(f"Loaded tabular data with shape {df.shape}")

    # Determine alignment size (min length across all modalities)
    min_len = min(len(wav_files), len(png_files), len(df))
    print(f"Aligning the first {min_len} rows across all modalities...")

    wav_files = wav_files[:min_len]
    png_files = png_files[:min_len]
    df = df.iloc[:min_len].reset_index(drop=True)

    # 2. Extract features
    print("\nExtracting Audio Features (MFCC)... This may take a couple of minutes.")
    audio_features = []
    for i, wav_path in enumerate(wav_files):
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{min_len} audio files...")
        audio_features.append(extract_audio_features(wav_path))

    print("\nExtracting Image Features (Flattened pixels)...")
    image_features = []
    for i, png_path in enumerate(png_files):
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{min_len} image files...")
        image_features.append(extract_image_features(png_path))

    # Create DataFrames for modalities
    audio_df = pd.DataFrame(audio_features, columns=[f'audio_mfcc_{i}' for i in range(13)])
    image_df = pd.DataFrame(image_features, columns=[f'img_pixel_{i}' for i in range(32*32)])

    # 3. Multimodal Fusion
    print("\n--- Phase 3: Multimodal Fusion ---")
    print("Constructing Mega-DataFrame...")
    
    # Separate target variable from the original DataFrame
    target_col = 'Mental_Health_Status'
    tabular_features = df.drop(columns=[target_col])
    target_series = df[target_col]

    # Concatenate tabular + audio + vision + target
    mega_df = pd.concat([tabular_features, audio_df, image_df, target_series], axis=1)
    print(f"Mega-DataFrame constructed successfully! Final shape: {mega_df.shape}")

    # 4. Preprocessing
    print("\n--- Phase 4: Preprocessing (Imputation, Scaling, Train-Test Split) ---")
    
    X = mega_df.drop(columns=[target_col])
    y = mega_df[target_col]

    # Imputation for any missing values across the mega dataframe
    print("Imputing missing values with median...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Encoding Target
    print("Encoding target labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-Test Split (80:20)
    print("Performing 80:20 Train-Test split...")
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)

    # Scaling Numerical features across the entire multimodal feature set
    print("Applying StandardScaler (fitted only on training set)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Output Processed Arrays
    print("\nSaving final preprocessed multimodal arrays to disk...")
    np.save('X_train_mega.npy', X_train_scaled)
    np.save('X_test_mega.npy', X_test_scaled)
    np.save('y_train_mega.npy', y_train)
    np.save('y_test_mega.npy', y_test)

    print("Saving preprocessor objects for GUI deployment...")
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    print("\n--- Multimodal Preprocessing Complete ---")
    print(f"X_train_mega shape: {X_train_scaled.shape}")
    print(f"X_test_mega shape:  {X_test_scaled.shape}")
    print(f"y_train_mega shape: {y_train.shape}")
    print(f"y_test_mega shape:  {y_test.shape}")

if __name__ == "__main__":
    main()
