"""
Script para gerar dataset sintético de raios X para demonstração
Este script cria imagens sintéticas que simulam raios X de tórax
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import random

def create_synthetic_xray(image_type='NORMAL', size=(224, 224)):
    """
    Cria uma imagem sintética de raio X
    
    Args:
        image_type: 'NORMAL' ou 'PNEUMONIA'
        size: Tamanho da imagem (largura, altura)
    
    Returns:
        PIL Image
    """
    # Criar imagem base com tons de cinza
    img = Image.new('L', size, color=30)
    draw = ImageDraw.Draw(img)
    
    # Adicionar estrutura de costelas (linhas horizontais curvas)
    for i in range(5, size[1]-5, 20):
        y_offset = random.randint(-5, 5)
        points = [(0, i+y_offset)]
        for x in range(0, size[0], 10):
            y = i + y_offset + int(10 * np.sin(x / 20))
            points.append((x, y))
        draw.line(points, fill=60, width=2)
    
    # Adicionar área pulmonar (elipses)
    # Pulmão esquerdo
    left_lung = [
        size[0]//4 - 40, size[1]//4,
        size[0]//2 - 10, size[1] - size[1]//4
    ]
    draw.ellipse(left_lung, fill=50, outline=70)
    
    # Pulmão direito
    right_lung = [
        size[0]//2 + 10, size[1]//4,
        3*size[0]//4 + 40, size[1] - size[1]//4
    ]
    draw.ellipse(right_lung, fill=50, outline=70)
    
    # Adicionar coluna vertebral (centro)
    spine = [
        size[0]//2 - 5, size[1]//6,
        size[0]//2 + 5, size[1] - size[1]//6
    ]
    draw.rectangle(spine, fill=80)
    
    if image_type == 'PNEUMONIA':
        # Adicionar opacidades (manchas brancas) para simular pneumonia
        num_opacities = random.randint(3, 7)
        for _ in range(num_opacities):
            x = random.randint(size[0]//4, 3*size[0]//4)
            y = random.randint(size[1]//4, 3*size[1]//4)
            radius = random.randint(15, 35)
            
            # Criar opacidade com gradiente
            for r in range(radius, 0, -2):
                opacity = int(120 + (radius - r) * 2)
                draw.ellipse(
                    [x-r, y-r, x+r, y+r],
                    fill=min(opacity, 200)
                )
        
        # Adicionar textura de infiltrado
        for _ in range(100):
            x = random.randint(0, size[0])
            y = random.randint(0, size[1])
            draw.point((x, y), fill=random.randint(100, 150))
    
    # Adicionar ruído
    img_array = np.array(img)
    noise = np.random.normal(0, 10, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Aplicar blur para suavizar
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img

def generate_dataset(base_path='data', num_samples_per_class=100):
    """
    Gera dataset sintético completo
    
    Args:
        base_path: Caminho base para salvar o dataset
        num_samples_per_class: Número de amostras por classe em cada split
    """
    splits = {
        'train': {'NORMAL': 300, 'PNEUMONIA': 300},
        'val': {'NORMAL': 50, 'PNEUMONIA': 50},
        'test': {'NORMAL': 100, 'PNEUMONIA': 100}
    }
    
    for split, classes in splits.items():
        for class_name, num_samples in classes.items():
            class_path = os.path.join(base_path, split, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            print(f"Gerando {num_samples} imagens para {split}/{class_name}...")
            
            for i in range(num_samples):
                img = create_synthetic_xray(image_type=class_name)
                img_path = os.path.join(class_path, f"{class_name}_{i:04d}.jpeg")
                img.save(img_path, quality=85)
            
            print(f"  ✓ {num_samples} imagens salvas em {class_path}")
    
    print("\n✓ Dataset sintético gerado com sucesso!")
    print(f"\nEstrutura do dataset:")
    print(f"  Train: 600 imagens (300 NORMAL + 300 PNEUMONIA)")
    print(f"  Val: 100 imagens (50 NORMAL + 50 PNEUMONIA)")
    print(f"  Test: 200 imagens (100 NORMAL + 100 PNEUMONIA)")

if __name__ == "__main__":
    print("="*60)
    print("Gerando Dataset Sintético de Raios X de Pneumonia")
    print("="*60)
    print("\nNOTA: Este é um dataset sintético para demonstração.")
    print("Para uso em produção, utilize o dataset real do Kaggle:")
    print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("="*60)
    print()
    
    generate_dataset()
