import gradio as gr
import numpy as np
import pandas as pd
import librosa
from PIL import Image
import joblib
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

# Define the PyTorch Multimodal CNN architecture
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
        tab_out = self.tab_fc(tab)
        aud_out = self.aud_fc(aud)
        img_out = self.cnn(img)
        merged = torch.cat((tab_out, aud_out, img_out), dim=1)
        return self.fusion(merged)

# Load models at startup
print("Loading CNN model artifacts for GUI...")
try:
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Load PyTorch weights
    cnn_model = MultimodalNet()
    cnn_model.load_state_dict(torch.load('multimodal_cnn.pth', weights_only=True))
    cnn_model.eval()
    
    # Load training data medians for missing tabular features
    df = pd.read_csv('cleaned_psychiatric_data.csv')
    tabular_medians = df.drop(columns=['Mental_Health_Status']).median(numeric_only=True)
except Exception as e:
    print(f"Warning: Models not fully loaded. Error: {e}")

hig_css = """
@import url('https://fonts.cdnfonts.com/css/sf-pro-display');

body, .gradio-container {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #F5F5F7 !important;
    color: #1D1D1F !important;
}

.gr-box, .gr-panel, .gr-form, .gr-block {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
}

h1, h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #1D1D1F !important;
}

.primary-btn {
    background-color: #007AFF !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 2px 5px rgba(0, 122, 255, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
}

.primary-btn:hover {
    background-color: #0062CC !important;
    transform: scale(0.98) !important;
}
"""

def predict_real(audio_path, image_path, depression, anxiety, stress, sleep):
    if not audio_path or not image_path:
        return "Error: Please upload both an audio and image sample."
    
    try:
        # 1. Tabular features (fill 4 provided, rest median)
        tabular_input = tabular_medians.copy()
        tabular_input['Depression_Score'] = depression
        tabular_input['Anxiety_Score'] = anxiety
        tabular_input['Stress_Score'] = stress
        tabular_input['Sleep_Quality'] = sleep
        tabular_input['Heart_Rate_BPM'] = 70 + (anxiety * 0.8)
        
        tab_array = tabular_input.values
        
        # 2. Audio features (13 MFCCs)
        y, sr = librosa.load(audio_path, sr=16000, duration=2.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        audio_array = np.mean(mfcc, axis=1)
        
        # 3. Image features (1024 pixels)
        img = Image.open(image_path).convert('L').resize((32, 32))
        img_array = np.array(img).flatten()
        
        # 4. Fusion and standard preprocessing (using the fitted models)
        fused_features = np.concatenate([tab_array, audio_array, img_array]).reshape(1, -1)
        fused_imputed = imputer.transform(fused_features)
        fused_scaled = scaler.transform(fused_imputed)
        
        # 5. Split back out and convert to PyTorch tensors
        tab_t = torch.tensor(fused_scaled[:, :21], dtype=torch.float32)
        aud_t = torch.tensor(fused_scaled[:, 21:34], dtype=torch.float32)
        img_t = torch.tensor(fused_scaled[:, 34:].reshape(-1, 1, 32, 32), dtype=torch.float32)
        
        # 6. CNN Inference
        with torch.no_grad():
            outputs = cnn_model(tab_t, aud_t, img_t)
            _, pred_idx = torch.max(outputs, 1)
        
        # 7. Decode
        pred_label = le.inverse_transform(pred_idx.numpy())[0]
        return f"Mental Health Status: {pred_label}"
    except Exception as e:
        return f"Inference Error: {str(e)}"

# Setup the UI
with gr.Blocks() as demo:
    gr.Markdown("# Deep Multimodal CNN Pipeline", elem_classes="text-center")
    gr.Markdown("Powered by PyTorch: Upload raw media to pass through fully decoupled CNN, Audio, and Tabular embedding branches.", elem_classes="text-center")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Raw Media Inputs")
            audio_in = gr.Audio(label="Voice Sample (.wav)", type="filepath")
            image_in = gr.Image(label="Facial Expression (.png)", type="filepath")
            
        with gr.Column(scale=1):
            gr.Markdown("### Tabular Metrics")
            dep_score = gr.Slider(0, 50, value=15, label="Depression Score")
            anx_score = gr.Slider(0, 50, value=20, label="Anxiety Score")
            stress_score = gr.Slider(0, 50, value=19, label="Stress Score")
            sleep_qual = gr.Slider(1, 10, value=5, label="Sleep Quality (1-10)")
            
            submit_btn = gr.Button("Run CNN Diagnostic", variant="primary", elem_classes="primary-btn")
            
            output = gr.Textbox(label="Predicted Mental Health Status", lines=2)
            
    submit_btn.click(
        fn=predict_real, 
        inputs=[audio_in, image_in, dep_score, anx_score, stress_score, sleep_qual], 
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False, css=hig_css, theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"))
