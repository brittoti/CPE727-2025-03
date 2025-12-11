"""
Detecção de Pneumonia usando Redes Neurais Convolucionais (CNN) com PyTorch

Este script implementa um modelo CNN completo com:
- Arquiteturas modernas (EfficientNet, DenseNet, ResNet)
- Técnicas de regularização (Dropout, Batch Normalization, Weight Decay, Data Augmentation)
- Métricas completas (Accuracy, Precision, Recall, F1, Specificity, AUC-ROC)
- Visualizações (Loss curves, Confusion Matrix, ROC Curve)

Uso:
    python cnn_pneumonia_detection.py --data_dir /caminho/para/dados

Referências:
[1] An, Q., Chen, W., & Shao, W. (2024). A Deep Convolutional Neural Network for 
    Pneumonia Detection in X-ray Images with Attention Ensemble. Diagnostics, 14(4), 390.
[2] Stephen, O., Sain, M., & Maduh, U. J. (2019). An efficient deep learning approach 
    to pneumonia classification in healthcare. Journal of Healthcare Engineering.
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
import argparse
import shutil
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from tqdm import tqdm
import json
import time
import sys

# Configurações padrão
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 15  # Ajustado para 15 épocas
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_CLASSES = 2


# ==================== DATASET CUSTOMIZADO ====================

class PneumoniaDataset(Dataset):
    """Dataset customizado para imagens de raios X de pneumonia"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Diretório raiz com subpastas NORMAL e PNEUMONIA
            transform: Transformações a serem aplicadas nas imagens
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Carregar todos os caminhos de imagens e labels
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
    """
    Retorna transformações apropriadas para cada split
    
    Data Augmentation para regularização:
    - Random rotation, flip, zoom
    - Color jitter para simular diferentes condições de imagem
    - Normalização com média e desvio padrão do ImageNet
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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


# ==================== MODELO CNN COM REGULARIZAÇÃO ====================

class PneumoniaCNN(nn.Module):
    """
    Modelo CNN híbrido para detecção de pneumonia
    
    Arquitetura:
    - Backbone: EfficientNet-B0 (pré-treinado no ImageNet)
    - Técnicas de regularização:
        * Dropout (0.5)
        * Batch Normalization
        * Weight Decay (via optimizer)
    - Attention mechanism para focar em regiões relevantes
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5, pretrained=True):
        super(PneumoniaCNN, self).__init__()
        
        # Backbone: EfficientNet-B0 pré-treinado
        # Usando weights em vez de pretrained (novo padrão do torchvision)
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Obter número de features do backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Remover classificador original
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism simples
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, num_features),
            nn.Sigmoid()
        )
        
        # Classificador customizado com regularização
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extração de features
        features = self.backbone(x)
        
        # Aplicar attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Classificação
        output = self.classifier(features)
        return output


class DenseNetPneumonia(nn.Module):
    """
    Modelo alternativo usando DenseNet121
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5, pretrained=True):
        super(DenseNetPneumonia, self).__init__()
        
        # Backbone: DenseNet121 pré-treinado
        if pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
            self.backbone = models.densenet121(weights=weights)
        else:
            self.backbone = models.densenet121(weights=None)
        num_features = self.backbone.classifier.in_features
        
        # Classificador customizado
        self.backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ==================== MÉTRICAS DE AVALIAÇÃO ====================

class MetricsCalculator:
    """Calcula todas as métricas de avaliação"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Calcula métricas completas de classificação
        
        Returns:
            dict com todas as métricas
        """
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Especificidade
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC-ROC (se probabilidades disponíveis)
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
        """Imprime métricas de forma formatada"""
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
    """Classe para gerenciar treinamento do modelo"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler=None, device=DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Histórico de treinamento
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self):
        """Treina por uma época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Treinamento')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Atualizar barra de progresso
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
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
        print(f"{'INICIANDO TREINAMENTO':^60}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nÉpoca {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Treinar
            train_loss, train_acc = self.train_epoch()
            
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
            
            # Salvar melhor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f"✓ Novo melhor modelo salvo! Val Acc: {val_acc:.4f}")
            
            # Imprimir resumo da época
            print(f"\nResumo da Época {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Carregar melhor modelo
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ Melhor modelo carregado (Val Acc: {self.best_val_acc:.4f})")
        
        return self.history


# ==================== AVALIAÇÃO ====================

def evaluate_model(model, test_loader, device=DEVICE):
    """
    Avalia o modelo no conjunto de teste
    
    Returns:
        y_true, y_pred, y_pred_proba, metrics
    """
    model.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Avaliação'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Probabilidades
            probs = torch.softmax(outputs, dim=1)
            
            # Predições
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(probs[:, 1].cpu().numpy())
    
    # Calcular métricas
    metrics = MetricsCalculator.calculate_metrics(
        y_true, y_pred, y_pred_proba
    )
    
    return np.array(y_true), np.array(y_pred), np.array(y_pred_proba), metrics


# ==================== VISUALIZAÇÕES ====================

def plot_training_history(history, save_path='results/cnn_training_history.png'):
    """Plota curvas de loss e acurácia"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Loss durante Treinamento', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Acurácia durante Treinamento', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Acurácia')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[2].set_title('Learning Rate', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de treinamento salvo em: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path='results/cnn_confusion_matrix.png'):
    """Plota matriz de confusão"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'],
                cbar_kws={'label': 'Contagem'})
    plt.title('Matriz de Confusão - CNN', fontsize=16, fontweight='bold')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Matriz de confusão salva em: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path='results/cnn_roc_curve.png'):
    """Plota curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC - CNN', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Curva ROC salva em: {save_path}")
    plt.close()


# ==================== ARGUMENTOS CLI ====================

def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description='Treinamento de CNN para Detecção de Pneumonia',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/leonardobrito/Documents/Deep Learning/Pneumonia Detection Using PyTorch with CNN and RNN Techniques/chest_xray',
        help='Diretório raiz contendo subpastas train, val, test'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Tamanho do batch para treinamento'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=NUM_EPOCHS,
        help='Número de épocas de treinamento'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate inicial'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='efficientnet',
        choices=['efficientnet', 'densenet'],
        help='Arquitetura do modelo'
    )
    
    return parser.parse_args()


# ==================== MAIN ====================

def main():
    """Função principal"""
    # Parse argumentos
    args = parse_args()
    
    # Atualizar configurações globais
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    
    print("\n" + "="*60)
    print("DETECÇÃO DE PNEUMONIA COM CNN - PyTorch".center(60))
    print("="*60)
    print(f"\nUsando dispositivo: {DEVICE}")
    print(f"PyTorch versão: {torch.__version__}")
    print(f"\nConfiguração:")
    print(f"  • Diretório de dados: {args.data_dir}")
    print(f"  • Batch size: {BATCH_SIZE}")
    print(f"  • Épocas: {NUM_EPOCHS}")
    print(f"  • Learning rate: {LEARNING_RATE}")
    print(f"  • Modelo: {args.model}")
    
    # Limpar cache do PyTorch em caso de corrupção
    print("\n[0/7] Limpando cache do PyTorch...")
    torch_cache = os.path.expanduser('~/.cache/torch/hub/checkpoints/')
    if os.path.exists(torch_cache):
        try:
            # Remove TODOS os arquivos do EfficientNet
            removed_files = []
            for file in os.listdir(torch_cache):
                if 'efficientnet' in file.lower() or file.startswith('efficientnet'):
                    file_path = os.path.join(torch_cache, file)
                    try:
                        os.remove(file_path)
                        removed_files.append(file)
                    except Exception as e:
                        print(f"  ⚠ Não foi possível remover {file}: {e}")
            
            if removed_files:
                print(f"  ✓ Removidos {len(removed_files)} arquivo(s) de cache")
            else:
                print(f"  ℹ Nenhum cache corrompido encontrado")
        except Exception as e:
            print(f"  ⚠ Aviso ao acessar cache: {e}")
    
    # Definir caminhos
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    
    # Verificar se os diretórios existem
    print("\n[1/7] Verificando caminhos dos dados...")
    for name, path in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} [NÃO ENCONTRADO]")
            sys.exit(1)
    
    # Criar diretórios de saída
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Carregar dados
    print("\n[2/7] Carregando dados...")
    train_dataset = PneumoniaDataset(train_dir, transform=get_transforms('train'))
    val_dataset = PneumoniaDataset(val_dir, transform=get_transforms('val'))
    test_dataset = PneumoniaDataset(test_dir, transform=get_transforms('test'))
    
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
    print(f"\n[3/7] Criando modelo CNN ({args.model.upper()})...")
    
    # Tentar baixar pesos com tratamento de erro
    try:
        print("  → Baixando/verificando pesos pré-treinados...")
        
        if args.model == 'efficientnet':
            # Tentar sem verificação de hash primeiro
            try:
                model = PneumoniaCNN(num_classes=NUM_CLASSES, dropout_rate=0.5, pretrained=True)
            except RuntimeError as e:
                if "invalid hash value" in str(e):
                    print("  ⚠ Hash inválido detectado. Limpando cache e tentando novamente...")
                    # Limpar cache específico do EfficientNet
                    torch_cache = os.path.expanduser('~/.cache/torch/hub/checkpoints/')
                    for file in os.listdir(torch_cache):
                        if 'efficientnet' in file.lower():
                            try:
                                os.remove(os.path.join(torch_cache, file))
                                print(f"    • Removido: {file}")
                            except:
                                pass
                    # Tentar novamente
                    model = PneumoniaCNN(num_classes=NUM_CLASSES, dropout_rate=0.5, pretrained=True)
                else:
                    raise
        else:
            model = DenseNetPneumonia(num_classes=NUM_CLASSES, dropout_rate=0.5, pretrained=True)
        
        print(f"  ✓ Modelo criado com {sum(p.numel() for p in model.parameters()):,} parâmetros")
        
    except Exception as e:
        print(f"  ✗ Erro ao criar modelo: {e}")
        print("\n" + "="*60)
        print("SOLUÇÃO ALTERNATIVA".center(60))
        print("="*60)
        print("\nExecute este comando no terminal para limpar o cache manualmente:")
        print("  rm -rf ~/.cache/torch/hub/checkpoints/*efficientnet*")
        print("\nOu tente usar o modelo DenseNet:")
        print(f"  python {sys.argv[0]} --model densenet")
        print("\n")
        sys.exit(1)
    
    # Definir loss, optimizer e scheduler
    print("\n[4/7] Configurando otimização...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    print(f"  ✓ Loss: CrossEntropyLoss")
    print(f"  ✓ Optimizer: Adam (lr={LEARNING_RATE}, weight_decay=1e-4)")
    print(f"  ✓ Scheduler: StepLR (step_size=7, gamma=0.1)")
    
    # Treinar
    print("\n[5/7] Treinando modelo...")
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler)
    history = trainer.train(NUM_EPOCHS)
    
    # Salvar modelo
    model_path = 'models/cnn_pneumonia_best.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Modelo salvo em: {model_path}")
    
    # Avaliar no conjunto de teste
    print("\n[6/7] Avaliando modelo no conjunto de teste...")
    y_true, y_pred, y_pred_proba, metrics = evaluate_model(model, test_loader)
    
    # Imprimir métricas
    MetricsCalculator.print_metrics(metrics, "MÉTRICAS FINAIS - CNN")
    
    # Salvar métricas em JSON
    metrics_to_save = {k: v for k, v in metrics.items() 
                       if k != 'confusion_matrix'}
    metrics_to_save['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    
    with open('results/cnn_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"✓ Métricas salvas em: results/cnn_metrics.json")
    
    # Criar visualizações
    print("\n[7/7] Gerando visualizações...")
    plot_training_history(history)
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(y_true, y_pred_proba)
    
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!".center(60))
    print("="*60)
    print("\nArquivos gerados:")
    print("  • models/cnn_pneumonia_best.pth")
    print("  • results/cnn_metrics.json")
    print("  • results/cnn_training_history.png")
    print("  • results/cnn_confusion_matrix.png")
    print("  • results/cnn_roc_curve.png")
    print("\n")


if __name__ == "__main__":
    main()