import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict

class MNISTDataHandler:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full MNIST dataset
        self.full_train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=self.transform
        )
        self.full_test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=self.transform
        )
        
        # Group indices by class
        self.train_indices_by_class = defaultdict(list)
        self.test_indices_by_class = defaultdict(list)
        
        for idx, (_, label) in enumerate(self.full_train_dataset):
            self.train_indices_by_class[label].append(idx)
            
        for idx, (_, label) in enumerate(self.full_test_dataset):
            self.test_indices_by_class[label].append(idx)
    
    def get_sample_images(self, num_per_class=5):
        """Get sample images for each class to display in GUI"""
        samples = {}
        for class_label in range(10):
            samples[class_label] = []
            indices = self.train_indices_by_class[class_label][:num_per_class]
            for idx in indices:
                image, _ = self.full_train_dataset[idx]
                samples[class_label].append(image.squeeze().numpy())
        return samples
    
    def create_custom_dataset(self, selected_classes, samples_per_class, train_ratio=0.7, val_ratio=0.15):
        """Create custom train/val/test datasets based on user selection (legacy method)"""
        train_indices = []
        val_indices = []
        test_indices = []
        
        for class_label, num_samples in zip(selected_classes, samples_per_class):
            if num_samples == 0:
                continue
                
            available_indices = self.train_indices_by_class[class_label]
            # Shuffle to get random samples
            np.random.shuffle(available_indices)
            
            # Take only the requested number of samples
            class_indices = available_indices[:num_samples]
            
            # Split into train/val/test
            num_train = int(len(class_indices) * train_ratio)
            num_val = int(len(class_indices) * val_ratio)
            
            train_indices.extend(class_indices[:num_train])
            val_indices.extend(class_indices[num_train:num_train + num_val])
            test_indices.extend(class_indices[num_train + num_val:])
        
        # Create datasets
        train_dataset = Subset(self.full_train_dataset, train_indices)
        val_dataset = Subset(self.full_train_dataset, val_indices)
        test_dataset = Subset(self.full_train_dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_custom_dataset_with_splits(self, selected_classes, train_samples, val_samples, test_samples):
        """Create custom train/val/test datasets with explicit split sizes"""
        train_indices = []
        val_indices = []
        test_indices = []
        
        for i, class_label in enumerate(selected_classes):
            train_count = train_samples[i]
            val_count = val_samples[i]
            test_count = test_samples[i]
            
            if train_count == 0 and val_count == 0 and test_count == 0:
                continue
                
            available_indices = self.train_indices_by_class[class_label].copy()
            # Shuffle to get random samples
            np.random.shuffle(available_indices)
            
            # Take samples for each split
            current_idx = 0
            
            # Training samples
            if train_count > 0:
                train_indices.extend(available_indices[current_idx:current_idx + train_count])
                current_idx += train_count
            
            # Validation samples
            if val_count > 0:
                val_indices.extend(available_indices[current_idx:current_idx + val_count])
                current_idx += val_count
            
            # Test samples
            if test_count > 0:
                test_indices.extend(available_indices[current_idx:current_idx + test_count])
        
        # Create datasets
        train_dataset = Subset(self.full_train_dataset, train_indices)
        val_dataset = Subset(self.full_train_dataset, val_indices)
        test_dataset = Subset(self.full_train_dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size=64):
        """Create data loaders from datasets"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader 