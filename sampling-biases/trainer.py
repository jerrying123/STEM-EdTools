import torch
import torch.nn.functional as F
import torch.optim as optim
from model import MNISTModel
import threading
import numpy as np

class ModelTrainer:
    def __init__(self, progress_callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.progress_callback = progress_callback
        self.training = False
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if not self.training:  # Allow stopping training
                break
                
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0 and self.progress_callback:
                progress = (batch_idx + 1) / len(train_loader)
                accuracy = 100. * correct / total
                self.progress_callback(f"Training: {progress:.1%} - Loss: {loss.item():.4f} - Acc: {accuracy:.2f}%")
        
        return train_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= total
        accuracy = 100. * correct / total
        return val_loss, accuracy
    
    def test(self, test_loader):
        """Test the model and return detailed results including confusion matrix"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        # Initialize confusion matrix
        confusion_matrix = np.zeros((10, 10), dtype=int)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Per-class accuracy and confusion matrix
                for i in range(target.size(0)):
                    true_label = target[i].item()
                    pred_label = pred[i].item()
                    class_total[true_label] += 1
                    confusion_matrix[true_label][pred_label] += 1
                    if pred_label == true_label:
                        class_correct[true_label] += 1
        
        test_loss /= total
        overall_accuracy = 100. * correct / total
        
        # Calculate per-class accuracies
        class_accuracies = {}
        for i in range(10):
            if class_total[i] > 0:
                class_accuracies[i] = 100. * class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0
        
        return test_loss, overall_accuracy, class_accuracies, class_total, confusion_matrix
    
    def train_model(self, train_loader, val_loader, epochs=5):
        """Train the model for specified epochs"""
        self.training = True
        
        def training_thread():
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            
            for epoch in range(epochs):
                if not self.training:
                    break
                    
                if self.progress_callback:
                    self.progress_callback(f"Epoch {epoch + 1}/{epochs}")
                
                # Train
                train_loss, train_acc = self.train_epoch(train_loader)
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                
                # Validate
                val_loss, val_acc = self.validate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                if self.progress_callback:
                    self.progress_callback(
                        f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                        f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
                        f"Val Acc: {val_acc:.2f}%"
                    )
            
            self.training = False
            if self.progress_callback:
                self.progress_callback("Training completed!")
        
        # Run training in a separate thread to avoid blocking GUI
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
        return thread
    
    def stop_training(self):
        """Stop training"""
        self.training = False 