import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """Combined loss function with perceptual and structural elements."""

    def __init__(self, alpha=0.6, beta=0.2, gamma=0.2):
        """
        Initialize combined loss.
        
        Args:
            alpha: L1 loss weight
            beta: Perceptual loss weight
            gamma: SSIM loss weight
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()

    def gradient_loss(self, pred, target):
        """Compute gradient loss for edge preservation."""
        def compute_gradient(img):
            padded = F.pad(img, (1, 1, 1, 1), mode='reflect')

            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32).reshape(1, 1, 3, 3).to(img.device)
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32).reshape(1, 1, 3, 3).to(img.device)

            grad_x = []
            grad_y = []
            for c in range(img.shape[1]):
                grad_x.append(F.conv2d(
                    padded[:, c:c+1], sobel_x, padding=0))
                grad_y.append(F.conv2d(
                    padded[:, c:c+1], sobel_y, padding=0))

            grad_x = torch.cat(grad_x, dim=1)
            grad_y = torch.cat(grad_y, dim=1)

            return grad_x, grad_y

        pred_grad_x, pred_grad_y = compute_gradient(pred)
        target_grad_x, target_grad_y = compute_gradient(target)

        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)

        return torch.mean(grad_diff_x) + torch.mean(grad_diff_y)

    def forward(self, pred, target, degradation_type=None):
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image
            target: Target clean image
            degradation_type: Type of degradation (unused in current implementation)
        """
        l1 = self.l1_loss(pred, target)

        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        pred_std = torch.std(pred, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])

        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)

        gradient_loss = self.gradient_loss(pred, target)

        perceptual_loss = mean_loss + std_loss + 0.5 * gradient_loss

        # SSIM loss (1 - SSIM to make it a loss to minimize)
        ssim_val = calculate_ssim(pred, target)
        ssim_loss = 1.0 - ssim_val

        # Combine all losses
        total_loss = (self.alpha * l1 +
                      self.beta * perceptual_loss +
                      self.gamma * ssim_loss)

        return total_loss


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (torch.Tensor): First image tensor
        img2 (torch.Tensor): Second image tensor
        
    Returns:
        float: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def gaussian(window_size, sigma):
    """
    Generate a gaussian kernel.
    
    Args:
        window_size (int): Size of the gaussian window
        sigma (float): Standard deviation of the gaussian
        
    Returns:
        torch.Tensor: Normalized gaussian kernel
    """
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-(x.pow(2.0) / (2.0 * sigma ** 2)))
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    """
    Create a gaussian window for SSIM calculation.
    
    Args:
        window_size (int): Size of the window
        channel (int): Number of channels
        sigma (float): Standard deviation for gaussian kernel
        
    Returns:
        torch.Tensor: 2D gaussian window expanded for all channels
    """
    _1d_window = gaussian(window_size, sigma).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(
        channel, 1, window_size, window_size
    ).contiguous()
    return window


def calculate_ssim(img1, img2, window_size=11, size_average=True, sigma=1.5):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (torch.Tensor): First image tensor
        img2 (torch.Tensor): Second image tensor
        window_size (int): Size of the gaussian window
        size_average (bool): Whether to average the result
        sigma (float): Standard deviation for gaussian kernel
        
    Returns:
        torch.Tensor: SSIM value(s)
    """
    # SSIM constants
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # Check if the inputs are batched
    batch_mode = len(img1.shape) == 4
    if not batch_mode:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    channel = img1.size(1)

    window = create_window(window_size, channel, sigma).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size//2, groups=channel
    ) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size//2, groups=channel
    ) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size//2, groups=channel
    ) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
