import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class IntelDataset(Dataset):
    """Классификация ландшафтов"""
    
    def __init__(
        self,
        root_dir: str,
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples: List[Tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size: int = 224, augment: bool = True):
    """Получить transforms для обучения/валидации"""
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_dataloaders(config: dict):
    """Создать DataLoader'ы для train/test"""
    train_transform = get_transforms(config['data']['img_size'], augment=True)
    test_transform = get_transforms(config['data']['img_size'], augment=False)
    
    train_dataset = IntelDataset(config['data']['train_dir'], transform=train_transform)
    test_dataset = IntelDataset(config['data']['test_dir'], transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, test_loader, train_dataset.classes