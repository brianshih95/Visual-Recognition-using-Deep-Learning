import os
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from torch.amp import GradScaler, autocast


# For reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# Constants
INPUT_DIR = 'vr-hw1/data'
OUTPUT_DIR = 'vr-hw1/output'
MODEL_NAME = 'resnet152'
BATCH_SIZE = 128
ACCUMULATION_STEPS = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'prediction.csv')
LOSS_FUNCTION = 'cross_entropy'

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class ImageDataset(Dataset):
    def __init__(self, root_dir, mode, transform):
        """
        Args:
            root_dir: Root directory containing the dataset
            mode: 'train', 'val', or 'test'
            transform: Image transformations to apply
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if mode in ['train', 'val']:
            mode_dir = os.path.join(root_dir, mode)
            for class_id in range(100):
                class_dir = os.path.join(mode_dir, str(class_id))
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(class_id)
        else:
            test_dir = os.path.join(root_dir, 'test')
            for img_name in os.listdir(test_dir):
                img_path = os.path.join(test_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(-1)  # No labels for test data

        print(f'Found {len(self.image_paths)} images for {mode} mode')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]

        if self.mode == 'test':
            return image, os.path.basename(img_path)
        else:
            return image, label


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.backbone = models.resnet152(weights='IMAGENET1K_V2')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def get_loss_function(loss_type, num_classes):
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(classes=num_classes)

    return nn.CrossEntropyLoss()


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs, accumulation_steps=ACCUMULATION_STEPS):
    """
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        accumulation_steps: Number of steps to accumulate gradients
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = GradScaler()
    best_val_acc = 0.0
    best_model_wts = model.state_dict()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        optimizer.zero_grad()

        batch_idx = 0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Mixed precision training
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss /= accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * inputs.size(0) * accumulation_steps

            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            batch_idx += 1
            if batch_idx % accumulation_steps == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        val_losses.append(val_loss)
        val_accs.append(val_acc.item())

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.7f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            print(f'New best model saved with accuracy: {best_val_acc:.4f}')

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    print(f'Best val Acc: {best_val_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, f'training_curves_{LOSS_FUNCTION}.png'))
    plt.show()

    print(
        f"Training curves saved to {os.path.join(OUTPUT_DIR, f'training_curves_{LOSS_FUNCTION}.png')}")


def predict(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for filename, pred in zip(filenames, preds):
                predictions.append((filename, pred.item()))

    return predictions


def main():
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImageDataset(
        INPUT_DIR, mode='train', transform=train_transform)
    val_dataset = ImageDataset(
        INPUT_DIR, mode='val', transform=test_transform)
    test_dataset = ImageDataset(
        INPUT_DIR, mode='test', transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = Classifier(num_classes=100)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')

    # Set up loss function
    criterion = get_loss_function(LOSS_FUNCTION, num_classes=100)
    print(f'Using loss function: {LOSS_FUNCTION}')

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=3,
        threshold=0.0001,
        min_lr=1e-7,
    )

    # Train model
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=NUM_EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS
    )

    # Save trained model
    model_save_path = os.path.join(OUTPUT_DIR, f'model_{LOSS_FUNCTION}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Generate predictions
    predictions = predict(model, test_loader)

    # Save predictions to CSV
    output_file = os.path.join(OUTPUT_DIR, f'prediction_{LOSS_FUNCTION}.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'pred_label'])
        for filename, label in predictions:
            writer.writerow([os.path.splitext(filename)[0], label])

    print(f'Predictions saved to {output_file}')


if __name__ == "__main__":
    main()
