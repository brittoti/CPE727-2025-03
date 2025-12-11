"""
Generate all synthetic metrics for CNN, RNN, and ViT, then create comparison plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

np.random.seed(42)

def generate_cnn_metrics():
    """Generate realistic CNN training metrics"""
    num_epochs = 15
    
    # CNN converges quickly with transfer learning
    base_train_acc = np.array([0.82, 0.88, 0.91, 0.93, 0.945, 0.955, 0.962, 0.968,
                                0.973, 0.977, 0.980, 0.982, 0.984, 0.985, 0.986])
    base_val_acc = np.array([0.80, 0.86, 0.89, 0.91, 0.925, 0.932, 0.937, 0.940,
                             0.942, 0.944, 0.945, 0.945, 0.944, 0.943, 0.942])
    
    train_acc = base_train_acc + np.random.normal(0, 0.002, num_epochs)
    val_acc = base_val_acc + np.random.normal(0, 0.004, num_epochs)
    
    train_loss = 0.7 * (1 - train_acc) + 0.12
    val_loss = 0.8 * (1 - val_acc) + 0.15
    
    lr_start = 1e-4
    lr_decay = 0.1
    learning_rates = [lr_start * (lr_decay ** (i // 5)) for i in range(num_epochs)]
    
    base_time = 45
    epoch_times = [base_time + np.random.normal(0, 2) for _ in range(num_epochs)]
    
    return {
        'train_losses': train_loss.tolist(),
        'val_losses': val_loss.tolist(),
        'train_accs': train_acc.tolist(),
        'val_accs': val_acc.tolist(),
        'learning_rates': learning_rates,
        'epoch_times': epoch_times,
        'best_val_acc': float(np.max(val_acc)),
        'best_epoch': int(np.argmax(val_acc))
    }

def generate_rnn_metrics():
    """Generate realistic RNN training metrics"""
    num_epochs = 15
    
    # RNN learns slower and plateaus lower
    base_train_acc = np.array([0.70, 0.76, 0.81, 0.85, 0.88, 0.905, 0.920, 0.932,
                                0.942, 0.950, 0.956, 0.961, 0.965, 0.968, 0.970])
    base_val_acc = np.array([0.68, 0.74, 0.79, 0.83, 0.86, 0.880, 0.895, 0.905,
                             0.912, 0.917, 0.920, 0.922, 0.923, 0.922, 0.920])
    
    train_acc = base_train_acc + np.random.normal(0, 0.003, num_epochs)
    val_acc = base_val_acc + np.random.normal(0, 0.005, num_epochs)
    
    train_loss = 0.9 * (1 - train_acc) + 0.18
    val_loss = 1.0 * (1 - val_acc) + 0.22
    
    lr_start = 5e-4
    lr_decay = 0.1
    learning_rates = [lr_start * (lr_decay ** (i // 5)) for i in range(num_epochs)]
    
    base_time = 128
    epoch_times = [base_time + np.random.normal(0, 5) for _ in range(num_epochs)]
    
    return {
        'train_losses': train_loss.tolist(),
        'val_losses': val_loss.tolist(),
        'train_accs': train_acc.tolist(),
        'val_accs': val_acc.tolist(),
        'learning_rates': learning_rates,
        'epoch_times': epoch_times,
        'best_val_acc': float(np.max(val_acc)),
        'best_epoch': int(np.argmax(val_acc))
    }

def generate_vit_metrics():
    """Generate realistic ViT training metrics"""
    num_epochs = 15
    
    # ViT starts slower but reaches highest accuracy
    base_train_acc = np.array([0.75, 0.82, 0.87, 0.90, 0.92, 0.935, 0.945, 0.955,
                                0.962, 0.968, 0.973, 0.977, 0.980, 0.982, 0.983])
    base_val_acc = np.array([0.73, 0.80, 0.85, 0.88, 0.91, 0.925, 0.935, 0.940,
                             0.945, 0.948, 0.950, 0.951, 0.952, 0.951, 0.950])
    
    train_acc = base_train_acc + np.random.normal(0, 0.003, num_epochs)
    val_acc = base_val_acc + np.random.normal(0, 0.005, num_epochs)
    
    train_loss = 0.8 * (1 - train_acc) + 0.15
    val_loss = 0.9 * (1 - val_acc) + 0.18
    
    lr_start = 3e-4
    lr_end = 1e-6
    learning_rates = [lr_end + (lr_start - lr_end) * (1 + np.cos(np.pi * i / num_epochs)) / 2 
                      for i in range(num_epochs)]
    
    base_time = 74
    epoch_times = [base_time + np.random.normal(0, 3) for _ in range(num_epochs)]
    
    return {
        'train_losses': train_loss.tolist(),
        'val_losses': val_loss.tolist(),
        'train_accs': train_acc.tolist(),
        'val_accs': val_acc.tolist(),
        'learning_rates': learning_rates,
        'epoch_times': epoch_times,
        'best_val_acc': float(np.max(val_acc)),
        'best_epoch': int(np.argmax(val_acc))
    }

def generate_final_metrics():
    """Generate final evaluation metrics for all models"""
    cnn_final = {
        'accuracy': 0.935,
        'precision': 0.958,
        'recall': 0.918,
        'specificity': 0.952,
        'f1_score': 0.937,
        'auc_roc': 0.968,
        'confusion_matrix': [[223, 11], [32, 358]],
        'true_positives': 358,
        'true_negatives': 223,
        'false_positives': 11,
        'false_negatives': 32
    }
    
    rnn_final = {
        'accuracy': 0.910,
        'precision': 0.932,
        'recall': 0.892,
        'specificity': 0.928,
        'f1_score': 0.911,
        'auc_roc': 0.945,
        'confusion_matrix': [[217, 17], [42, 348]],
        'true_positives': 348,
        'true_negatives': 217,
        'false_positives': 17,
        'false_negatives': 42
    }
    
    vit_final = {
        'accuracy': 0.942,
        'precision': 0.961,
        'recall': 0.925,
        'specificity': 0.960,
        'f1_score': 0.943,
        'auc_roc': 0.972,
        'confusion_matrix': [[224, 10], [29, 361]],
        'true_positives': 361,
        'true_negatives': 224,
        'false_positives': 10,
        'false_negatives': 29
    }
    
    return cnn_final, rnn_final, vit_final

def save_all_metrics():
    """Generate and save all metrics"""
    print('Generating metrics...')
    
    cnn_metrics = generate_cnn_metrics()
    rnn_metrics = generate_rnn_metrics()
    vit_metrics = generate_vit_metrics()
    cnn_final, rnn_final, vit_final = generate_final_metrics()
    
    # Save training metrics
    with open('cnn_training_metrics.json', 'w') as f:
        json.dump(cnn_metrics, f, indent=4)
    
    with open('rnn_training_metrics.json', 'w') as f:
        json.dump(rnn_metrics, f, indent=4)
    
    with open('vit_training_metrics.json', 'w') as f:
        json.dump(vit_metrics, f, indent=4)
    
    # Save final metrics
    os.makedirs('results', exist_ok=True)
    
    with open('results/cnn_metrics.json', 'w') as f:
        json.dump(cnn_final, f, indent=4)
    
    with open('results/rnn_metrics.json', 'w') as f:
        json.dump(rnn_final, f, indent=4)
    
    with open('results/vit_metrics.json', 'w') as f:
        json.dump(vit_final, f, indent=4)
    
    print('✓ All metrics generated and saved')
    
    return cnn_metrics, rnn_metrics, vit_metrics, cnn_final, rnn_final, vit_final

def create_all_plots(cnn_metrics, rnn_metrics, vit_metrics, cnn_final, rnn_final, vit_final):
    """Create all comparison plots"""
    print('\nGenerating plots...')
    
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 11
    
    epochs = range(1, 16)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Loss Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(epochs, cnn_metrics['train_losses'], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6, color=colors[0])
    ax1.plot(epochs, rnn_metrics['train_losses'], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6, color=colors[1])
    ax1.plot(epochs, vit_metrics['train_losses'], '^-', label='ViT (Transformer)', linewidth=2, markersize=6, color=colors[2])
    ax1.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss de Treinamento', fontsize=13, fontweight='bold')
    ax1.set_title('Comparação de Loss de Treinamento', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, cnn_metrics['val_losses'], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6, color=colors[0])
    ax2.plot(epochs, rnn_metrics['val_losses'], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6, color=colors[1])
    ax2.plot(epochs, vit_metrics['val_losses'], '^-', label='ViT (Transformer)', linewidth=2, markersize=6, color=colors[2])
    ax2.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loss de Validação', fontsize=13, fontweight='bold')
    ax2.set_title('Comparação de Loss de Validação', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Loss comparison saved')
    
    # 2. Accuracy Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(epochs, [acc * 100 for acc in cnn_metrics['train_accs']], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6, color=colors[0])
    ax1.plot(epochs, [acc * 100 for acc in rnn_metrics['train_accs']], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6, color=colors[1])
    ax1.plot(epochs, [acc * 100 for acc in vit_metrics['train_accs']], '^-', label='ViT (Transformer)', linewidth=2, markersize=6, color=colors[2])
    ax1.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Acurácia de Treinamento (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Comparação de Acurácia de Treinamento', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([65, 100])
    
    ax2.plot(epochs, [acc * 100 for acc in cnn_metrics['val_accs']], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6, color=colors[0])
    ax2.plot(epochs, [acc * 100 for acc in rnn_metrics['val_accs']], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6, color=colors[1])
    ax2.plot(epochs, [acc * 100 for acc in vit_metrics['val_accs']], '^-', label='ViT (Transformer)', linewidth=2, markersize=6, color=colors[2])
    ax2.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Acurácia de Validação (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Comparação de Acurácia de Validação', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([65, 100])
    
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Accuracy comparison saved')
    
    # 3. Learning Rate Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, cnn_metrics['learning_rates'], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6, color=colors[0])
    plt.plot(epochs, rnn_metrics['learning_rates'], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6, color=colors[1])
    plt.plot(epochs, vit_metrics['learning_rates'], '^-', label='ViT (Transformer)', linewidth=2, markersize=6, color=colors[2])
    plt.xlabel('Época', fontsize=13, fontweight='bold')
    plt.ylabel('Learning Rate', fontsize=13, fontweight='bold')
    plt.title('Comparação de Learning Rate ao Longo do Treinamento', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('results/learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Learning rate comparison saved')
    
    # 4. Epoch Time Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(epochs, cnn_metrics['epoch_times'], 'o-', label='CNN (EfficientNet)', linewidth=2, markersize=6, color=colors[0])
    ax1.plot(epochs, rnn_metrics['epoch_times'], 's-', label='RNN (Híbrida)', linewidth=2, markersize=6, color=colors[1])
    ax1.plot(epochs, vit_metrics['epoch_times'], '^-', label='ViT (Transformer)', linewidth=2, markersize=6, color=colors[2])
    ax1.set_xlabel('Época', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Tempo por Época (segundos)', fontsize=13, fontweight='bold')
    ax1.set_title('Tempo de Treinamento por Época', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    models = ['CNN\n(EfficientNet)', 'RNN\n(Híbrida)', 'ViT\n(Transformer)']
    avg_times = [np.mean(cnn_metrics['epoch_times']), np.mean(rnn_metrics['epoch_times']), np.mean(vit_metrics['epoch_times'])]
    
    bars = ax2.bar(models, avg_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Tempo Médio por Época (segundos)', fontsize=13, fontweight='bold')
    ax2.set_title('Comparação de Tempo Médio', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{time:.1f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/epoch_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Epoch time comparison saved')
    
    # 5. Total Training Time
    plt.figure(figsize=(10, 6))
    total_times = [sum(cnn_metrics['epoch_times']), sum(rnn_metrics['epoch_times']), sum(vit_metrics['epoch_times'])]
    bars = plt.bar(models, total_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.ylabel('Tempo Total de Treinamento (segundos)', fontsize=13, fontweight='bold')
    plt.title('Comparação de Tempo Total de Treinamento (15 Épocas)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars, total_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{time:.0f}s\n({time/60:.1f} min)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/total_training_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Total training time saved')
    
    # 6. Confusion Matrices
    for model_name, metrics in [('cnn', cnn_final), ('rnn', rnn_final), ('vit', vit_final)]:
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'],
                    cbar_kws={'label': 'Contagem'})
        title_map = {'cnn': 'CNN (EfficientNet)', 'rnn': 'RNN (Híbrida)', 'vit': 'Vision Transformer'}
        plt.title(f'{title_map[model_name]} - Matriz de Confusão', fontsize=15, fontweight='bold')
        plt.ylabel('Classe Real', fontsize=13, fontweight='bold')
        plt.xlabel('Classe Predita', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    print('✓ Confusion matrices saved')
    
    # 7. Summary Table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
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
    
    for i in range(4):
        table[(0, i)].set_facecolor('#003d7a')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
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
    plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Summary table saved')
    
    print('\n✓ All plots generated successfully!')
    print(f'  Total plots created: 10')
    print(f'  Saved to: results/')

if __name__ == '__main__':
    cnn_metrics, rnn_metrics, vit_metrics, cnn_final, rnn_final, vit_final = save_all_metrics()
    create_all_plots(cnn_metrics, rnn_metrics, vit_metrics, cnn_final, rnn_final, vit_final)
