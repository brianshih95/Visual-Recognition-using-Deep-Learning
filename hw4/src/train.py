import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils import calculate_psnr


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    scheduler):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        scheduler: Learning rate scheduler
        
    Returns:
        tuple: Average training loss and PSNR
    """
    model.train()
    total_loss = 0
    total_psnr = 0

    for degraded, clean, deg_type in tqdm(train_loader, desc='Training'):
        degraded = degraded.to(device)
        clean = clean.to(device)
        deg_type = deg_type.to(device)

        optimizer.zero_grad()
        output = model(degraded, deg_type)

        loss = criterion(output, clean, deg_type)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update learning rate each batch with OneCycleLR
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            psnr = calculate_psnr(output, clean)

        total_loss += loss.item()
        total_psnr += psnr.item()

    return total_loss / len(train_loader), total_psnr / len(train_loader)


def train_detector(detector, train_loader, val_loader, criterion, num_epochs,
                   device, save_path):
    """
    Train the degradation type detector.
    
    Args:
        detector: The detector model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        num_epochs (int): Number of training epochs
        device: Device to run training on
        save_path: Path to the saved detector
        
    Returns:
        DegradationTypeDetector: Trained detector model
    """
    optimizer = optim.AdamW(detector.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        detector.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training phase
        desc = f"Detector Train Epoch {epoch+1}/{num_epochs}"
        for degraded_imgs, _, degradation_types in tqdm(train_loader,
                                                        desc=desc):
            degraded_imgs = degraded_imgs.to(device)
            degradation_types = degradation_types.to(
                device).float().unsqueeze(1)

            outputs = detector(degraded_imgs)
            loss = criterion(outputs, degradation_types)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            train_total += degradation_types.size(0)
            train_correct += (predicted == degradation_types).sum().item()

        # Validation phase
        detector.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            desc = f"Detector Val Epoch {epoch+1}/{num_epochs}"
            for degraded_imgs, _, degradation_types in tqdm(val_loader,
                                                            desc=desc):
                degraded_imgs = degraded_imgs.to(device)
                degradation_types = (degradation_types.to(device)
                                     .float().unsqueeze(1))

                outputs = detector(degraded_imgs)
                loss = criterion(outputs, degradation_types)

                val_loss += loss.item()

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                val_total += degradation_types.size(0)
                val_correct += (predicted == degradation_types).sum().item()

        # Update learning rate
        scheduler.step()

        # Calculate metrics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        print(f"Detector Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_loss/len(train_loader):.4f}, "
              f"Accuracy: {train_accuracy:.2f}%")
        print(f"Val - Loss: {val_loss/len(val_loader):.4f}, "
              f"Accuracy: {val_accuracy:.2f}%")

        # Save best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(detector.state_dict(), save_path)
            print(f"New best detector saved with accuracy: "
                  f"{best_accuracy:.2f}%")

    # Load best model for inference
    detector.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"Best detector loaded with accuracy: {best_accuracy:.2f}%")

    return detector
