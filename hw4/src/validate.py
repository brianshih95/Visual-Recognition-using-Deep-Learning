import torch
from tqdm import tqdm

from utils import calculate_psnr


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        
    Returns:
        tuple: Average validation loss and PSNR
    """
    model.eval()
    total_loss = 0
    total_psnr = 0

    with torch.no_grad():
        for degraded, clean, deg_type in tqdm(val_loader, desc='Validation'):
            degraded = degraded.to(device)
            clean = clean.to(device)
            deg_type = deg_type.to(device)

            output = model(degraded, deg_type)

            loss = criterion(output, clean, deg_type)
            psnr = calculate_psnr(output, clean)

            total_loss += loss.item()
            total_psnr += psnr.item()

    return total_loss / len(val_loader), total_psnr / len(val_loader)
