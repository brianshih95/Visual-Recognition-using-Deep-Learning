import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import DegradedImageDataset
from model import PromptIR, DegradationTypeDetector
from utils import CombinedLoss
from train import train_one_epoch, train_detector
from validate import validate


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_predictions(model, detector, test_loader, device, save_path):
    """
    Generate predictions using the trained model and detector.
    
    Args:
        model: The trained restoration model
        detector: The trained degradation detector
        test_loader: Test data loader
        device: Device to run inference on
        save_path (str): Path to save predictions
    """
    model.eval()
    detector.eval()
    predictions = {}

    with torch.no_grad():
        for degraded_imgs, filenames in tqdm(test_loader,
                                             desc="Generating predictions"):
            degraded_imgs = degraded_imgs.to(device)

            # Detect degradation type
            degradation_types = detector(degraded_imgs)

            # Generate restored images with the original degradation type
            outputs1 = model(degraded_imgs, degradation_types)

            # For ensemble effect, also try with different degradation types
            # This can help in ambiguous degradation detection cases
            degradation_types_alt = torch.clamp(degradation_types + 0.2,
                                                0.0, 1.0)
            outputs2 = model(degraded_imgs, degradation_types_alt)

            degradation_types_alt2 = torch.clamp(degradation_types - 0.2,
                                                 0.0, 1.0)
            outputs3 = model(degraded_imgs, degradation_types_alt2)

            # Weighted average for final output
            # (giving more weight to original prediction)
            outputs = outputs1 * 0.8 + outputs2 * 0.1 + outputs3 * 0.1

            # Test-time augmentation - try horizontal flip
            degraded_imgs_flipped = torch.flip(degraded_imgs, [3])
            outputs_flipped = model(degraded_imgs_flipped, degradation_types)
            outputs_flipped = torch.flip(outputs_flipped, [3])

            # Average with flipped results
            outputs = (outputs + outputs_flipped) / 2.0

            # Final clipping to ensure values are in valid range
            outputs = torch.clamp(outputs, 0.0, 1.0)

            # Convert to numpy arrays and store
            for i, filename in enumerate(filenames):
                img = outputs[i].cpu().numpy().transpose(1, 2, 0) * 255.0
                img = img.astype(np.uint8)
                # Convert to CHW format for npz storage
                img = img.transpose(2, 0, 1)
                predictions[filename] = img

    # Save predictions to npz file
    np.savez(save_path, **predictions)
    print(f"Predictions saved to {save_path}")


def visualize_restored_images(model, detector, test_loader, device, num_samples, save_path):
    """Visualize restored images."""
    model.eval()
    detector.eval()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    sample_count = 0
    with torch.no_grad():
        for degraded_imgs, filenames in test_loader:
            if sample_count >= num_samples:
                break

            degraded_imgs = degraded_imgs.to(device)

            # Detect degradation type
            degradation_types = detector(degraded_imgs)

            # Generate restored images
            outputs = model(degraded_imgs, degradation_types)

            # Convert tensors to numpy arrays for visualization
            for i in range(min(len(degraded_imgs), num_samples - sample_count)):
                # Restored image
                restored_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
                restored_img = np.clip(restored_img, 0, 1)

                # Plot image
                axes[sample_count].imshow(restored_img)
                axes[sample_count].set_title(
                    f'Restored Image {sample_count+1}')
                axes[sample_count].axis('off')

                sample_count += 1

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_learning_curves(train_losses, val_losses, train_psnrs, val_psnrs, save_path):
    """Plot learning curves."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_psnrs, label='Train PSNR', color='blue')
    plt.plot(val_psnrs, label='Validation PSNR', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main training and inference pipeline."""
    # Data path
    data_root = '../data'

    # Data preprocessing with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = DegradedImageDataset(
        data_root, mode='train', transform=transform, augment=True)
    val_dataset = DegradedImageDataset(
        data_root, mode='val', transform=transform, augment=False)
    test_dataset = DegradedImageDataset(
        data_root, mode='test', transform=transform, augment=False)

    batch_size = 16

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PromptIR().to(device)

    # Combined loss function
    criterion = CombinedLoss()

    base_lr = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr,
                            betas=(0.9, 0.999), weight_decay=1e-4)

    num_epochs = 50

    # OneCycleLR scheduler with cosine annealing
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,  # Warm up for 10% of training
        div_factor=25,  # Initial learning rate via max_lr/div_factor
        # Min LR via max_lr/(div_factor*final_div_factor)
        final_div_factor=1000,
        anneal_strategy='cos'
    )

    best_psnr = 0
    best_epoch = 0
    early_stop_counter = 0
    patience = 5

    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []

    # Training loop with early stopping
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.8f}')

        # Train and validate
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler)
        val_loss, val_psnr = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnrs.append(train_psnr)
        val_psnrs.append(val_psnr)

        # Print results
        print(f'Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}dB')
        print(f'Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}dB')

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), '../assets/best_model.pth')
            print(f'New best model saved! PSNR: {best_psnr:.2f}dB '
                  f'(Epoch {best_epoch})')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f'No improvement for {early_stop_counter} epochs '
                  f'(Best PSNR: {best_psnr:.2f}dB at epoch {best_epoch})')

        # Early stopping
        if early_stop_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break

    # plot learning curves
    plot_learning_curves(train_losses, val_losses, train_psnrs, val_psnrs, 
                        save_path='../assets/learning_curves.png')

    # Clear GPU memory before training detector
    torch.cuda.empty_cache()

    # Initialize and train improved degradation type detector
    print("\n=== Training Degradation Type Detector ===")
    detector = DegradationTypeDetector().to(device)
    detector_criterion = nn.BCELoss()

    # Train detector with validation
    detector = train_detector(detector, train_loader, val_loader,
                              detector_criterion, num_epochs=10, device=device, save_path='../assets/best_detector.pth')

    # Load best model for prediction
    print("\n=== Loading Best Model for Prediction ===")
    model.load_state_dict(torch.load(
        '../assets/best_model.pth', weights_only=True))

    generate_predictions(model, detector, test_loader, device, save_path='../assets/pred.npz')
    
    # After generating predictions, visualize restored images
    visualize_restored_images(model, detector, test_loader, device, 10, save_path='../assets/restored_images')

    print("All tasks completed successfully!")


if __name__ == '__main__':
    main()
