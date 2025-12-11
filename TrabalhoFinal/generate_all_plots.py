"""
Comprehensive Visualization Script for Model Comparison
Generates all plots for CNN, RNN, and ViT models
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_loss_comparison(cnn_metrics, rnn_metrics, vit_metrics, save_path='results/loss_comparison.png'):
    """Plot training and validation loss for all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_cnn = range(1, len(cnn_metrics['train_losses']) + 1)
    epochs_rnn = range(1, len(rnn_metrics['train_losses']) + 1)
    epochs_vit = range(1, len(vit_metrics['train_losses']) + 1)
    
    # Training Loss
    ax1.plot(epochs_cnn, cnn_metrics['train_losses'], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6)
    ax1.plot(epochs_rnn, rnn_metrics['train_losses'], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6)
    ax1.plot(epochs_vit, vit_metrics['train_losses'], '^-', label='ViT (Transformer)', linewidth=2, markersize=6)
    ax1.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss de Treinamento', fontsize=13, fontweight='bold')
    ax1.set_title('Comparação de Loss de Treinamento', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    ax2.plot(epochs_cnn, cnn_metrics['val_losses'], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6)
    ax2.plot(epochs_rnn, rnn_metrics['val_losses'], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6)
    ax2.plot(epochs_vit, vit_metrics['val_losses'], '^-', label='ViT (Transformer)', linewidth=2, markersize=6)
    ax2.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loss de Validação', fontsize=13, fontweight='bold')
    ax2.set_title('Comparação de Loss de Validação', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Loss comparison plot saved to {save_path}')

def plot_accuracy_comparison(cnn_metrics, rnn_metrics, vit_metrics, save_path='results/accuracy_comparison.png'):
    """Plot training and validation accuracy for all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_cnn = range(1, len(cnn_metrics['train_accs']) + 1)
    epochs_rnn = range(1, len(rnn_metrics['train_accs']) + 1)
    epochs_vit = range(1, len(vit_metrics['train_accs']) + 1)
    
    # Training Accuracy
    ax1.plot(epochs_cnn, [acc * 100 for acc in cnn_metrics['train_accs']], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6)
    ax1.plot(epochs_rnn, [acc * 100 for acc in rnn_metrics['train_accs']], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6)
    ax1.plot(epochs_vit, [acc * 100 for acc in vit_metrics['train_accs']], '^-', label='ViT (Transformer)', linewidth=2, markersize=6)
    ax1.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Acurácia de Treinamento (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Comparação de Acurácia de Treinamento', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([70, 100])
    
    # Validation Accuracy
    ax2.plot(epochs_cnn, [acc * 100 for acc in cnn_metrics['val_accs']], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6)
    ax2.plot(epochs_rnn, [acc * 100 for acc in rnn_metrics['val_accs']], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6)
    ax2.plot(epochs_vit, [acc * 100 for acc in vit_metrics['val_accs']], '^-', label='ViT (Transformer)', linewidth=2, markersize=6)
    ax2.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Acurácia de Validação (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Comparação de Acurácia de Validação', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([70, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Accuracy comparison plot saved to {save_path}')

def plot_learning_rate_comparison(cnn_metrics, rnn_metrics, vit_metrics, save_path='results/learning_rate_comparison.png'):
    """Plot learning rate schedules for all models"""
    plt.figure(figsize=(12, 6))
    
    epochs_cnn = range(1, len(cnn_metrics['learning_rates']) + 1)
    epochs_rnn = range(1, len(rnn_metrics['learning_rates']) + 1)
    epochs_vit = range(1, len(vit_metrics['learning_rates']) + 1)
    
    plt.plot(epochs_cnn, cnn_metrics['learning_rates'], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6)
    plt.plot(epochs_rnn, rnn_metrics['learning_rates'], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6)
    plt.plot(epochs_vit, vit_metrics['learning_rates'], '^-', label='ViT (Transformer)', linewidth=2, markersize=6)
    
    plt.xlabel('Época', fontsize=13, fontweight='bold')
    plt.ylabel('Learning Rate', fontsize=13, fontweight='bold')
    plt.title('Comparação de Learning Rate ao Longo do Treinamento', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Learning rate comparison plot saved to {save_path}')

def plot_epoch_time_comparison(cnn_metrics, rnn_metrics, vit_metrics, save_path='results/epoch_time_comparison.png'):
    """Plot training time per epoch for all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_cnn = range(1, len(cnn_metrics['epoch_times']) + 1)
    epochs_rnn = range(1, len(rnn_metrics['epoch_times']) + 1)
    epochs_vit = range(1, len(vit_metrics['epoch_times']) + 1)
    
    # Time per epoch
    ax1.plot(epochs_cnn, cnn_metrics['epoch_times'], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6)
    ax1.plot(epochs_rnn, rnn_metrics['epoch_times'], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6)
    ax1.plot(epochs_vit, vit_metrics['epoch_times'], '^-', label='ViT (Transformer)', linewidth=2, markersize=6)
    ax1.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Tempo por Época (segundos)', fontsize=13, fontweight='bold')
    ax1.set_title('Tempo de Treinamento por Época', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Average time comparison (bar plot)
    models = ['CNN\n(EfficientNet)', 'RNN\n(Híbrida)', 'ViT\n(Transformer)']
    avg_times = [
        np.mean(cnn_metrics['epoch_times']),
        np.mean(rnn_metrics['epoch_times']),
        np.mean(vit_metrics['epoch_times'])
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax2.bar(models, avg_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Tempo Médio por Época (segundos)', fontsize=13, fontweight='bold')
    ax2.set_title('Comparação de Tempo Médio de Treinamento', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Epoch time comparison plot saved to {save_path}')

def plot_total_training_time(cnn_metrics, rnn_metrics, vit_metrics, save_path='results/total_training_time.png'):
    """Plot total training time comparison"""
    plt.figure(figsize=(10, 6))
    
    models = ['CNN\n(EfficientNet)', 'RNN\n(Híbrida)', 'ViT\n(Transformer)']
    total_times = [
        sum(cnn_metrics['epoch_times']),
        sum(rnn_metrics['epoch_times']),
        sum(vit_metrics['epoch_times'])
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = plt.bar(models, total_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.ylabel('Tempo Total de Treinamento (segundos)', fontsize=13, fontweight='bold')
    plt.title('Comparação de Tempo Total de Treinamento (15 Épocas)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, total_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.0f}s\n({time/60:.1f} min)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Total training time plot saved to {save_path}')

def plot_convergence_analysis(cnn_metrics, rnn_metrics, vit_metrics, save_path='results/convergence_analysis.png'):
    """Analyze convergence speed (epochs to reach 90% accuracy)"""
    plt.figure(figsize=(12, 6))
    
    epochs_cnn = range(1, len(cnn_metrics['val_accs']) + 1)
    epochs_rnn = range(1, len(rnn_metrics['val_accs']) + 1)
    epochs_vit = range(1, len(vit_metrics['val_accs']) + 1)
    
    # Plot validation accuracy
    plt.plot(epochs_cnn, [acc * 100 for acc in cnn_metrics['val_accs']], 'o-', label='CNN (EfficientNet)', linewidth=2.5, markersize=8)
    plt.plot(epochs_rnn, [acc * 100 for acc in rnn_metrics['val_accs']], 's-', label='RNN (Híbrida)', linewidth=2.5, markersize=8)
    plt.plot(epochs_vit, [acc * 100 for acc in vit_metrics['val_accs']], '^-', label='ViT (Transformer)', linewidth=2.5, markersize=8)
    
    # Add 90% threshold line
    plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Threshold 90%', alpha=0.7)
    
    plt.xlabel('Época', fontsize=13, fontweight='bold')
    plt.ylabel('Acurácia de Validação (%)', fontsize=13, fontweight='bold')
    plt.title('Análise de Convergência - Velocidade de Aprendizado', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([70, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Convergence analysis plot saved to {save_path}')

def plot_overfitting_analysis(cnn_metrics, rnn_metrics, vit_metrics, save_path='results/overfitting_analysis.png'):
    """Analyze overfitting (gap between train and validation accuracy)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models_data = [
        ('CNN (EfficientNet)', cnn_metrics, 'o-'),
        ('RNN (Híbrida)', rnn_metrics, 's-'),
        ('ViT (Transformer)', vit_metrics, '^-')
    ]
    
    for ax, (name, metrics, marker) in zip(axes, models_data):
        epochs = range(1, len(metrics['train_accs']) + 1)
        train_accs = [acc * 100 for acc in metrics['train_accs']]
        val_accs = [acc * 100 for acc in metrics['val_accs']]
        gap = [t - v for t, v in zip(train_accs, val_accs)]
        
        ax.plot(epochs, train_accs, marker, label='Treino', linewidth=2, markersize=6, color='#1f77b4')
        ax.plot(epochs, val_accs, marker, label='Validação', linewidth=2, markersize=6, color='#ff7f0e')
        ax.fill_between(epochs, train_accs, val_accs, alpha=0.2, color='red', label=f'Gap (média: {np.mean(gap):.2f}%)')
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Acurácia (%)', fontsize=11, fontweight='bold')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([70, 100])
    
    plt.suptitle('Análise de Overfitting - Gap entre Treino e Validação', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Overfitting analysis plot saved to {save_path}')

def create_summary_table(cnn_metrics, rnn_metrics, vit_metrics, cnn_final, rnn_final, vit_final, save_path='results/summary_table.png'):
    """Create a comprehensive summary table"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    metrics_names = [
        'Acurácia (%)',
        'Precisão (%)',
        'Sensibilidade (%)',
        'Especificidade (%)',
        'F1-Score (%)',
        'AUC-ROC',
        '',
        'Parâmetros (M)',
        'Tempo/Época (s)',
        'Tempo Total (min)',
        'Melhor Época',
        'Melhor Val Acc (%)'
    ]
    
    cnn_values = [
        f"{cnn_final['accuracy']*100:.2f}",
        f"{cnn_final['precision']*100:.2f}",
        f"{cnn_final['recall']*100:.2f}",
        f"{cnn_final['specificity']*100:.2f}",
        f"{cnn_final['f1_score']*100:.2f}",
        f"{cnn_final['auc_roc']:.4f}",
        '',
        '4.2',
        f"{np.mean(cnn_metrics['epoch_times']):.1f}",
        f"{sum(cnn_metrics['epoch_times'])/60:.1f}",
        f"{cnn_metrics['best_epoch'] + 1}",
        f"{cnn_metrics['best_val_acc']*100:.2f}"
    ]
    
    rnn_values = [
        f"{rnn_final['accuracy']*100:.2f}",
        f"{rnn_final['precision']*100:.2f}",
        f"{rnn_final['recall']*100:.2f}",
        f"{rnn_final['specificity']*100:.2f}",
        f"{rnn_final['f1_score']*100:.2f}",
        f"{rnn_final['auc_roc']:.4f}",
        '',
        '9.1',
        f"{np.mean(rnn_metrics['epoch_times']):.1f}",
        f"{sum(rnn_metrics['epoch_times'])/60:.1f}",
        f"{rnn_metrics['best_epoch'] + 1}",
        f"{rnn_metrics['best_val_acc']*100:.2f}"
    ]
    
    vit_values = [
        f"{vit_final['accuracy']*100:.2f}",
        f"{vit_final['precision']*100:.2f}",
        f"{vit_final['recall']*100:.2f}",
        f"{vit_final['specificity']*100:.2f}",
        f"{vit_final['f1_score']*100:.2f}",
        f"{vit_final['auc_roc']:.4f}",
        '',
        '86.5',
        f"{np.mean(vit_metrics['epoch_times']):.1f}",
        f"{sum(vit_metrics['epoch_times'])/60:.1f}",
        f"{vit_metrics['best_epoch'] + 1}",
        f"{vit_metrics['best_val_acc']*100:.2f}"
    ]
    
    table_data = []
    for i, metric in enumerate(metrics_names):
        table_data.append([metric, cnn_values[i], rnn_values[i], vit_values[i]])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Métrica', 'CNN\n(EfficientNet)', 'RNN\n(Híbrida)', 'ViT\n(Transformer)'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.35, 0.22, 0.22, 0.22])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#003d7a')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows
    for i in range(1, len(metrics_names) + 1):
        if metrics_names[i-1] == '':
            for j in range(4):
                table[(i, j)].set_facecolor('#e0e0e0')
        else:
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f5f5f5')
                else:
                    table[(i, j)].set_facecolor('white')
        
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Tabela Comparativa Completa - CNN vs RNN vs ViT', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Summary table saved to {save_path}')

def main():
    print('Loading metrics...')
    
    # Load training metrics
    cnn_metrics = load_metrics('cnn_training_metrics.json')
    rnn_metrics = load_metrics('rnn_training_metrics.json')
    vit_metrics = load_metrics('vit_training_metrics.json')
    
    # Load final evaluation metrics
    cnn_final = load_metrics('results/cnn_metrics.json')
    rnn_final = load_metrics('results/rnn_metrics.json')
    vit_final = load_metrics('results/vit_metrics.json')
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print('\nGenerating plots...')
    
    # Generate all plots
    plot_training_loss_comparison(cnn_metrics, rnn_metrics, vit_metrics)
    plot_accuracy_comparison(cnn_metrics, rnn_metrics, vit_metrics)
    plot_learning_rate_comparison(cnn_metrics, rnn_metrics, vit_metrics)
    plot_epoch_time_comparison(cnn_metrics, rnn_metrics, vit_metrics)
    plot_total_training_time(cnn_metrics, rnn_metrics, vit_metrics)
    plot_convergence_analysis(cnn_metrics, rnn_metrics, vit_metrics)
    plot_overfitting_analysis(cnn_metrics, rnn_metrics, vit_metrics)
    create_summary_table(cnn_metrics, rnn_metrics, vit_metrics, cnn_final, rnn_final, vit_final)
    
    print('\n✓ All plots generated successfully!')
    print(f'  Total plots created: 8')
    print(f'  Saved to: results/')

if __name__ == '__main__':
    main()
