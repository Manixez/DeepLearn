import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import time
import os
from torchvision.models import mnasnet0_5, MNASNet0_5_Weights  # Changed to MNASNet

# Import our custom modules
from datareader import MakananIndo
from utils import check_set_gpu


def create_label_encoder(dataset):
    """Create a mapping from string labels to numeric indices"""
    all_labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        all_labels.append(label)
    
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    return label_to_idx, idx_to_label, unique_labels


def create_mnasnet_model(num_classes, freeze_backbone=True):
    """Create MNASNet0.5 model with transfer learning"""
    # Load pretrained MNASNet model
    weights = MNASNet0_5_Weights.IMAGENET1K_V1
    model = mnasnet0_5(weights=weights)
    
    # Print model info
    print(f"Loaded MNASNet0.5 pretrained model with weights: {weights}")
    print(f"Original classifier: {model.classifier}")
    
    # Freeze all parameters if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("Froze all backbone parameters")
    
    # Replace the final classification layer
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.Softmax(dim=1)
    )
    
    print(f"Replaced classifier with new layer: {model.classifier}")
    print(f"New classifier output features: {num_classes}")
    
    # Ensure the new classifier parameters are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model


def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate accuracy, F1-score, precision, and recall"""
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multiclass classification, use 'weighted' average for imbalanced datasets
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1, precision, recall


def train_epoch(model, train_loader, criterion, optimizer, device, label_to_idx):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_idx, batch_data in enumerate(train_loader):
        images, labels_tuple, filepath = batch_data
        
        # Convert string labels to numeric indices
        if isinstance(labels_tuple, tuple):
            # Convert string labels to indices
            label_indices = [label_to_idx[label] for label in labels_tuple]
            labels = torch.tensor(label_indices, dtype=torch.long)
        else:
            labels = labels_tuple
            
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_predictions, len(label_to_idx))
    
    return avg_loss, accuracy, f1, precision, recall


def validate_epoch(model, val_loader, criterion, device, label_to_idx):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            images, labels_tuple, _ = batch_data
            
            # Convert string labels to numeric indices
            if isinstance(labels_tuple, tuple):
                # Convert string labels to indices
                label_indices = [label_to_idx[label] for label in labels_tuple]
                labels = torch.tensor(label_indices, dtype=torch.long)
            else:
                labels = labels_tuple
                
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_predictions, len(label_to_idx))
    
    return avg_loss, accuracy, f1, precision, recall, all_labels, all_predictions


def main():
    # Set up device using utils function
    device = check_set_gpu()
    
    # Hyperparameters
    batch_size = 16  # Smaller batch size for larger images
    learning_rate = 8e-4
    num_epochs = 13  # Fewer epochs since we're using pretrained model
    img_size = 224  # Standard ImageNet size for ConvNeXt
    
    print(f"Using image size: {img_size}x{img_size}")
    
    # Create datasets with larger image size
    print("Loading datasets...")
    train_dataset = MakananIndo(split='train', img_size=(img_size, img_size))
    val_dataset = MakananIndo(split='val', img_size=(img_size, img_size))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create label encoder
    print("Creating label encoder...")
    label_to_idx, idx_to_label, unique_labels = create_label_encoder(train_dataset)
    num_classes = len(unique_labels)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {unique_labels}")
    print(f"Label to index mapping: {label_to_idx}")
    
    cpu_count = os.cpu_count()
    nworkers = cpu_count - 4
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nworkers)
    
    # Initialize MNASNet model with transfer learning
    print("\nInitializing MNASNet0.5 model...")
    model = create_mnasnet_model(num_classes, freeze_backbone=True)
    model = model.to(device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Loss function and optimizer (only optimize trainable parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    print(f"\nStarting training with:")
    print(f"- Device: {device}")
    print(f"- Model: MNASNet0.5 (pretrained)")
    print(f"- Image size: {img_size}x{img_size}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Number of epochs: {num_epochs}")
    print(f"- Transfer learning: Backbone frozen, classifier trainable")
    print("-" * 80)
    
    # Training loop
    best_val_accuracy = 0.0
    best_model_path = "best_mnasnet_model.pth"
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 50)
        
        start_time = time.time()
        
        # Training phase
        train_loss, train_acc, train_f1, train_precision, train_recall = train_epoch(
            model, train_loader, criterion, optimizer, device, label_to_idx
        )
        
        # Validation phase
        val_loss, val_acc, val_f1, val_precision, val_recall, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device, label_to_idx
        )
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
        
        print("-" * 80)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    
    # Final validation evaluation with detailed classification report
    val_loss, val_acc, val_f1, val_precision, val_recall, val_labels, val_preds = validate_epoch(
        model, val_loader, criterion, device, label_to_idx
    )
    
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    print(f"Final Validation Metrics:")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print("-" * 50)
    class_names = sorted(unique_labels)
    print(classification_report(val_labels, val_preds, target_names=[str(cls) for cls in class_names]))
    
    print(f"\nBest model saved as: {best_model_path}")
    
    # Print model summary
    print(f"\nModel Summary:")
    print(f"- Architecture: MNASNet0.5 with ImageNet pretraining")
    print(f"- Transfer Learning: Frozen backbone + trainable classifier")
    print(f"- Input size: 224x224x3")  # RegNet default input size
    print(f"- Output classes: {num_classes}")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
