# Hack4Health: Deep Multimodal Diagnostic Pipeline

This repository contains a full state-of-the-art multimodal machine learning pipeline for detecting Mental Health Status (Healthy, Mild Stress, Moderate Stress, Severe Stress).

## Architecture
The system utilizes a custom **PyTorch Convolutional Neural Network (CNN)** that evaluates three distinct modalities simultaneously:
1. **Visual Data**: Raw facial images are processed through 2D Convolutions to extract spatial topology (eyes, mouth, brow tension).
2. **Audio Data**: Voice samples are converted to Mel-Frequency Cepstral Coefficients (MFCCs) and processed via a Dense branch.
3. **Tabular Data**: Physiological and behavioral metrics (Depression, Anxiety, Sleep Quality, etc.) are processed via a Dense branch.

The embeddings are fused and run through a multi-class classifier.

## Quickstart

To run the interactive Apple HIG-styled Web GUI locally:

1. Ensure requirements are installed:
```bash
pip install torch torchvision gradio pandas numpy librosa Pillow scikit-learn
```

2. Run the Web Server:
```bash
python3 4_gui.py
```
3. Open your browser to `http://127.0.0.1:7860`.

## Pre-Trained Weights
The `.pth` PyTorch weights and `.pkl` scikit-learn preprocessors are included in this repository. You do not need the original raw datasets to launch the inference GUI.
