"""
Vision Transformer (ViT) for Pneumonia Detection from Chest X-Rays
Implements ViT-Base/16 architecture with comprehensive metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import time
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ChestXRayDataset(Dataset):
    """Custom Dataset for Chest X-Ray Images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load NORMAL images
        normal_dir = os.path.join(root_dir, 'NORMAL')
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)  # 0 = Normal
        
        # Load PNEUMONIA images
        pneumonia_dir = os.path.join(root_dir, 'PNEUMONIA')
        if os.path.exists(pneumonia_dir):
            for img_name in os.listdir(pneumonia_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(pneumonia_dir, img_name))
                    self.labels.append(1)  # 1 = Pneumonia
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class VisionTransformerPneumonia(nn.Module):
    """Vision Transformer for Pneumonia Classification"""
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(VisionTransformerPneumonia, self).__init__()
        
        # Load pretrained ViT-B/16 from torchvision
        # Note: torchvision.models.vit_b_16 requires torchvision >= 0.13
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            if pretrained:
                self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                self.vit = vit_b_16(weights=None)
        except ImportError:
            # Fallback: Create custom ViT-like architecture
            print("Warning: Using custom ViT implementation (torchvision ViT not available)")
            self.vit = self._create_custom_vit()
        
        # Get the dimension of ViT output
        vit_output_dim = 768  # ViT-B/16 hidden dimension
        
        # Replace classification head
        self.vit.heads = nn.Identity()  # Remove original head
        
        # Custom classification head with regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_output_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(vit_output_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def _create_custom_vit(self):
        """Fallback: Create a simplified ViT-like model using ResNet backbone"""
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        return model
    
    def forward(self, x):
        features = self.vit(x)
        output = self.classifier(features)
        return output

class MetricsTracker:
    """Track and store training metrics"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.epoch_times = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def update(self, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.train_losses) - 1
    
    def save(self, filepath):
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    from tqdm import tqdm
    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    from tqdm import tqdm
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

def evaluate_model(model, dataloader, device, save_dir='results'):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    # AUC-ROC
    auc_roc = roc_auc_score(all_labels, all_probs)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'auc_roc': float(auc_roc),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    # Save metrics
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'vit_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title('Vision Transformer - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ViT (AUC = {auc_roc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Vision Transformer - ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_roc_curve.png'), dpi=300)
    plt.close()
    
    return metrics

def main():
    # Configuration
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.3
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data paths
    DATA_DIR = '/Users/leonardobrito/Documents/Deep Learning/Pneumonia Detection Using PyTorch with CNN and RNN Techniques/chest_xray'
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'chest_xray', 'test')
    
    print(f'\nData directories:')
    print(f'  Train: {train_dir}')
    print(f'  Val:   {val_dir}')
    print(f'  Test:  {test_dir}')
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ChestXRayDataset(train_dir, transform=train_transform)
    val_dataset = ChestXRayDataset(val_dir, transform=val_transform)
    test_dataset = ChestXRayDataset(test_dir, transform=test_transform)
    
    # Calculate class weights for imbalanced dataset
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f'\nDataset sizes:')
    print(f'  Training samples: {len(train_dataset)}')
    print(f'  Validation samples: {len(val_dataset)}')
    print(f'  Test samples: {len(test_dataset)}')
    print(f'  Class weights: {class_weights}')
    
    # Create dataloaders (reduced num_workers to avoid hanging)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    # Initialize model
    print('\nInitializing Vision Transformer...')
    model = VisionTransformerPneumonia(num_classes=2, pretrained=True, dropout_rate=DROPOUT_RATE)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Metrics tracker
    tracker = MetricsTracker()
    
    # Training loop
    print('\nStarting training...')
    print(f'Total batches per epoch: {len(train_loader)}')
    total_start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print(f'{"="*60}')
        epoch_start_time = time.time()
        
        # Train
        print('Training phase...')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'✓ Training completed - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        
        # Validate
        print('Validation phase...')
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        print(f'✓ Validation completed - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Track metrics
        epoch_time = time.time() - epoch_start_time
        tracker.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
        
        # Print progress
        print(f'\nEpoch Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        print(f'  Time: {epoch_time:.2f}s')
        
        # Save best model
        if val_acc >= tracker.best_val_acc:
            torch.save(model.state_dict(), 'vit_best_model.pth')
            print(f'  ✓ Best model saved (Val Acc: {val_acc:.4f})')
    
    total_time = time.time() - total_start_time
    print(f'\nTotal training time: {total_time:.2f}s ({total_time/60:.2f} minutes)')
    print(f'Average time per epoch: {total_time/NUM_EPOCHS:.2f}s')
    
    # Save training metrics
    tracker.save('vit_training_metrics.json')
    
    # Final evaluation
    print('\nEvaluating on test set...')
    model.load_state_dict(torch.load('vit_best_model.pth'))
    final_metrics = evaluate_model(model, test_loader, device, save_dir='results')
    
    print('\n=== Final Test Metrics ===')
    print(f'Accuracy: {final_metrics["accuracy"]:.4f}')
    print(f'Precision: {final_metrics["precision"]:.4f}')
    print(f'Recall (Sensitivity): {final_metrics["recall"]:.4f}')
    print(f'Specificity: {final_metrics["specificity"]:.4f}')
    print(f'F1-Score: {final_metrics["f1_score"]:.4f}')
    print(f'AUC-ROC: {final_metrics["auc_roc"]:.4f}')
    print(f'\nConfusion Matrix:')
    print(f'TN: {final_metrics["true_negatives"]}, FP: {final_metrics["false_positives"]}')
    print(f'FN: {final_metrics["false_negatives"]}, TP: {final_metrics["true_positives"]}')

if __name__ == '__main__':
    main()