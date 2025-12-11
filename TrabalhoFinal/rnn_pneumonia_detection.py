"""
Detecção de Pneumonia usando Redes Neurais Recorrentes (RNN/LSTM) com PyTorch

Este script implementa um modelo híbrido CNN+LSTM para detecção de pneumonia:
- CNN para extração de features espaciais
- LSTM para análise sequencial de features
- Técnicas de regularização (Dropout recorrente, Weight Decay, Gradient Clipping)
- Métricas completas (Accuracy, Precision, Recall, F1, Specificity, AUC-ROC)

Referências:
[1] Shin, H. C., et al. (2016). Learning to Read Chest X-Rays: Recurrent Neural 
    Cascade Model for Automated Image Annotation. IEEE CVPR.
[2] Dey, S., et al. (2022). CovidConvLSTM: A fuzzy ensemble model for COVID-19 
    detection using chest X-ray images. Expert Systems with Applications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from tqdm import tqdm
import json

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16  # Menor batch size para RNN (mais memória)
NUM_EPOCHS = 15
LEARNING_RATE = 0.0005  # Learning rate menor para RNN
IMG_SIZE = 224
NUM_CLASSES = 2
SEQUENCE_LENGTH = 16  # Número de patches/regiões a processar sequencialmente

print(f"Usando dispositivo: {DEVICE}")
print(f"PyTorch versão: {torch.__version__}")


# ==================== DATASET CUSTOMIZADO ====================

class PneumoniaDataset(Dataset):
    """Dataset customizado para imagens de raios X de pneumonia"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpeg', '.jpg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== DATA AUGMENTATION ====================

def get_transforms(split='train'):
    """Retorna transformações apropriadas para cada split"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# ==================== MODELO CNN + LSTM ====================

class CNNFeatureExtractor(nn.Module):
    """
    CNN para extração de features espaciais
    Usa ResNet18 pré-treinado como backbone
    """
    
    def __init__(self, pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        
        # Backbone: ResNet18 pré-treinado
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remover camadas finais (avg pool e fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature dimension: 512 channels para ResNet18
        self.feature_dim = 512
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, H, W)
        Returns:
            features: (batch, 512, h, w) onde h=7, w=7 para input 224x224
        """
        features = self.features(x)
        return features


class PneumoniaLSTM(nn.Module):
    """
    Modelo híbrido CNN + LSTM para detecção de pneumonia
    
    Arquitetura:
    1. CNN extrai feature maps da imagem
    2. Feature maps são divididos em sequência de patches
    3. LSTM processa sequência de patches
    4. Classificador final usa último hidden state do LSTM
    
    Técnicas de regularização:
    - Dropout recorrente no LSTM
    - Dropout nas camadas densas
    - Batch Normalization
    - Gradient Clipping (aplicado no treinamento)
    """
    
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2, 
                 dropout_rate=0.3, pretrained=True):
        super(PneumoniaLSTM, self).__init__()
        
        # CNN para extração de features
        self.cnn = CNNFeatureExtractor(pretrained=pretrained)
        
        # Dimensões
        self.feature_dim = self.cnn.feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM bidirecional com dropout recorrente
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism para LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classificador com regularização
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, H, W)
        Returns:
            output: (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Extração de features com CNN
        # features: (batch, 512, 7, 7)
        features = self.cnn(x)
        
        # Reorganizar features em sequência
        # (batch, 512, 7, 7) -> (batch, 49, 512)
        features = features.view(batch_size, self.feature_dim, -1)
        features = features.permute(0, 2, 1)  # (batch, seq_len, feature_dim)
        
        # Processar com LSTM
        # lstm_out: (batch, seq_len, hidden_size*2)
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Aplicar attention mechanism
        # attention_weights: (batch, seq_len, 1)
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum usando attention
        # context: (batch, hidden_size*2)
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classificação
        output = self.classifier(context)
        
        return output


class ConvLSTMCell(nn.Module):
    """
    Célula ConvLSTM que mantém estrutura espacial
    Alternativa mais avançada ao LSTM tradicional
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Portas do LSTM usando convoluções
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenar input e hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Aplicar convolução
        combined_conv = self.conv(combined)
        
        # Split em 4 portas
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Aplicar ativações
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        # Atualizar cell state e hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class PneumoniaConvLSTM(nn.Module):
    """
    Modelo usando ConvLSTM para manter estrutura espacial
    Mais avançado que CNN+LSTM tradicional
    """
    
    def __init__(self, num_classes=2, hidden_dim=128, num_layers=2, 
                 dropout_rate=0.3, pretrained=True):
        super(PneumoniaConvLSTM, self).__init__()
        
        # CNN para extração inicial de features
        self.cnn = CNNFeatureExtractor(pretrained=pretrained)
        
        # ConvLSTM layers
        self.convlstm_cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=self.cnn.feature_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim
            ) for i in range(num_layers)
        ])
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extração de features
        features = self.cnn(x)  # (batch, 512, 7, 7)
        
        # Inicializar hidden e cell states
        h, w = features.size(2), features.size(3)
        h_states = []
        c_states = []
        
        for i in range(self.num_layers):
            h_dim = self.hidden_dim
            h_states.append(torch.zeros(batch_size, h_dim, h, w).to(x.device))
            c_states.append(torch.zeros(batch_size, h_dim, h, w).to(x.device))
        
        # Processar com ConvLSTM
        current_input = features
        for layer_idx in range(self.num_layers):
            h_states[layer_idx], c_states[layer_idx] = self.convlstm_cells[layer_idx](
                current_input, (h_states[layer_idx], c_states[layer_idx])
            )
            current_input = h_states[layer_idx]
        
        # Global pooling
        pooled = self.global_pool(h_states[-1])  # (batch, hidden_dim, 1, 1)
        pooled = pooled.view(batch_size, -1)     # (batch, hidden_dim)
        
        # Classificação
        output = self.classifier(pooled)
        
        return output


# ==================== MÉTRICAS DE AVALIAÇÃO ====================

class MetricsCalculator:
    """Calcula todas as métricas de avaliação"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        auc_score = None
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = auc(fpr, tpr)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc': auc_score,
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, title="Métricas de Avaliação"):
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"Acurácia (Accuracy):        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precisão (Precision):       {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall (Sensibilidade):     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:                   {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"Especificidade:             {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        if metrics['auc'] is not None:
            print(f"AUC-ROC:                    {metrics['auc']:.4f}")
        print(f"\nMatriz de Confusão:")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"{'='*60}\n")


# ==================== TREINAMENTO ====================

class Trainer:
    """Classe para gerenciar treinamento do modelo RNN"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler=None, device=DEVICE, gradient_clip=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip  # Gradient clipping para RNN
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'gradient_norms': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self):
        """Treina por uma época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []
        
        pbar = tqdm(self.train_loader, desc='Treinamento')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (importante para RNNs)
            if self.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                gradient_norms.append(grad_norm.item())
            
            self.optimizer.step()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0
        
        return epoch_loss, epoch_acc, avg_grad_norm
    
    def validate(self):
        """Valida o modelo"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validação'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs):
        """Treina o modelo por múltiplas épocas"""
        print(f"\n{'='*60}")
        print(f"{'INICIANDO TREINAMENTO - RNN':^60}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nÉpoca {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Treinar
            train_loss, train_acc, grad_norm = self.train_epoch()
            
            # Validar
            val_loss, val_acc = self.validate()
            
            # Atualizar learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Salvar histórico
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            self.history['gradient_norms'].append(grad_norm)
            
            # Salvar melhor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f"✓ Novo melhor modelo salvo! Val Acc: {val_acc:.4f}")
            
            # Imprimir resumo
            print(f"\nResumo da Época {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Avg Gradient Norm: {grad_norm:.4f}")
        
        # Carregar melhor modelo
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ Melhor modelo carregado (Val Acc: {self.best_val_acc:.4f})")
        
        return self.history


# ==================== AVALIAÇÃO ====================

def evaluate_model(model, test_loader, device=DEVICE):
    """Avalia o modelo no conjunto de teste"""
    model.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Avaliação'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(probs[:, 1].cpu().numpy())
    
    metrics = MetricsCalculator.calculate_metrics(
        y_true, y_pred, y_pred_proba
    )
    
    return np.array(y_true), np.array(y_pred), np.array(y_pred_proba), metrics


# ==================== VISUALIZAÇÕES ====================

def plot_training_history(history, save_path='results/rnn_training_history.png'):
    """Plota curvas de treinamento"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss durante Treinamento', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Acurácia durante Treinamento', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Acurácia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient Norms
    axes[1, 1].plot(epochs, history['gradient_norms'], 'm-', linewidth=2)
    axes[1, 1].set_title('Gradient Norm (Clipping)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de treinamento salvo em: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path='results/rnn_confusion_matrix.png'):
    """Plota matriz de confusão"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'],
                cbar_kws={'label': 'Contagem'})
    plt.title('Matriz de Confusão - RNN (CNN+LSTM)', fontsize=16, fontweight='bold')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Matriz de confusão salva em: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path='results/rnn_roc_curve.png'):
    """Plota curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='green', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC - RNN (CNN+LSTM)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Curva ROC salva em: {save_path}")
    plt.close()


# ==================== MAIN ====================

def main():
    """Função principal"""
    print("\n" + "="*60)
    print("DETECÇÃO DE PNEUMONIA COM RNN - PyTorch".center(60))
    print("="*60)
    
    # Criar diretórios
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Carregar dados
    print("\n[1/6] Carregando dados...")
    train_dataset = PneumoniaDataset('data/train', transform=get_transforms('train'))
    val_dataset = PneumoniaDataset('data/val', transform=get_transforms('val'))
    test_dataset = PneumoniaDataset('data/test', transform=get_transforms('test'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    print(f"  ✓ Train: {len(train_dataset)} imagens")
    print(f"  ✓ Val: {len(val_dataset)} imagens")
    print(f"  ✓ Test: {len(test_dataset)} imagens")
    
    # Criar modelo
    print("\n[2/6] Criando modelo RNN (CNN + Bidirectional LSTM)...")
    model = PneumoniaLSTM(
        num_classes=NUM_CLASSES, 
        hidden_size=256, 
        num_layers=2, 
        dropout_rate=0.3, 
        pretrained=True
    )
    print(f"  ✓ Modelo criado com {sum(p.numel() for p in model.parameters()):,} parâmetros")
    
    # Definir loss, optimizer e scheduler
    print("\n[3/6] Configurando otimização...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    print(f"  ✓ Loss: CrossEntropyLoss")
    print(f"  ✓ Optimizer: Adam (lr={LEARNING_RATE}, weight_decay=1e-4)")
    print(f"  ✓ Scheduler: StepLR (step_size=5, gamma=0.1)")
    print(f"  ✓ Gradient Clipping: 1.0")
    
    # Treinar
    print("\n[4/6] Treinando modelo...")
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, 
                     scheduler, gradient_clip=1.0)
    history = trainer.train(NUM_EPOCHS)
    
    # Salvar modelo
    model_path = 'models/rnn_pneumonia_best.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Modelo salvo em: {model_path}")
    
    # Avaliar no conjunto de teste
    print("\n[5/6] Avaliando modelo no conjunto de teste...")
    y_true, y_pred, y_pred_proba, metrics = evaluate_model(model, test_loader)
    
    # Imprimir métricas
    MetricsCalculator.print_metrics(metrics, "MÉTRICAS FINAIS - RNN")
    
    # Salvar métricas em JSON
    metrics_to_save = {k: v for k, v in metrics.items() 
                       if k != 'confusion_matrix'}
    metrics_to_save['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    
    with open('results/rnn_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"✓ Métricas salvas em: results/rnn_metrics.json")
    
    # Criar visualizações
    print("\n[6/6] Gerando visualizações...")
    plot_training_history(history)
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(y_true, y_pred_proba)
    
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!".center(60))
    print("="*60)
    print("\nArquivos gerados:")
    print("  • models/rnn_pneumonia_best.pth")
    print("  • results/rnn_metrics.json")
    print("  • results/rnn_training_history.png")
    print("  • results/rnn_confusion_matrix.png")
    print("  • results/rnn_roc_curve.png")
    print("\n")


if __name__ == "__main__":
    main()
