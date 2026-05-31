import os
import yaml
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.classifier import LandscapeClassifier
from src.data.dataset import get_dataloaders


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def train(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, test_loader, classes = get_dataloaders(config)
    print(f"Classes: {classes}")
    
    # Model
    model = LandscapeClassifier(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Этап 1: заморозить backbone, обучить только классификатор
    model.freeze_backbone()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['training']['scheduler_step'],
        gamma=config['training']['scheduler_gamma']
    )
    
    best_acc = 0.0
    os.makedirs('artifacts', exist_ok=True)
    
    print("\n=== Stage 1: Training classifier head ===")
    for epoch in range(config['training']['epochs'] // 2):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'artifacts/model_best.pth')
    
    # Этап 2: fine-tuning всей сети
    print("\n=== Stage 2: Fine-tuning entire network ===")
    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'] / 10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(config['training']['epochs'] // 2, config['training']['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, optimizer, device)
        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'artifacts/model_best.pth')
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    print("Model saved to artifacts/model_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_config.yaml')
    args = parser.parse_args()
    train(args.config)