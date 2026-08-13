import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Training on device: {device}")

    # 1. Data Augmentation & Loading
    # Resize to 224x224 for ResNet-18 optimal transfer learning
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # Convert 1-channel to 3-channel tensor
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root='archive/train', transform=train_transform)
    test_dataset = datasets.ImageFolder(root='archive/test', transform=test_transform)

    # Use smaller batch size to avoid out of memory on MPS
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 2. Architecture: Load ResNet-18
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # 3. Fine-Tuning Strategy: Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 4. Adapt Output: Replace the final layer to output 7 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7) # automatically requires_grad=True

    model = model.to(device)

    # 5. Training Loop setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    num_epochs = 10
    best_acc = 0.0
    best_model_wts = model.state_dict()

    print("Starting Transfer Learning on ResNet-18...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation evaluation to track best weights
        model.eval()
        corrects = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
        
        val_acc = corrects.float() / len(test_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    # Load best weights to run final evaluation
    model.load_state_dict(best_model_wts)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print("\n" + "="*40)
    print(" FINAL TRANSFER LEARNING METRICS ")
    print("="*40)
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Macro F1      : {f1 * 100:.2f}%")
    print("="*40)

    # 6. Save best weights
    torch.save(best_model_wts, 'fer_resnet18_best.pth')
    print("Saved best model to 'fer_resnet18_best.pth'")

if __name__ == '__main__':
    main()
