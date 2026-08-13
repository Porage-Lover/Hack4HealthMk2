import torch
import torch.nn as nn

class MultimodalNet(nn.Module):
    def __init__(self):
        super(MultimodalNet, self).__init__()
        # Deeper fully connected branches
        self.tab_fc = nn.Sequential(nn.Linear(21, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2))
        self.aud_fc = nn.Sequential(nn.Linear(13, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2))
        
        # Deep VGG-style CNN for 32x32 grayscale
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 4)
        )

    def forward(self, tab, aud, img):
        # Only flatten/concat what's needed
        if tab.size(0) == 1:
            # Handle batch_norm single batch bug during inference by eval mode
            pass
        return self.fusion(torch.cat((self.tab_fc(tab), self.aud_fc(aud), self.cnn(img)), dim=1))
