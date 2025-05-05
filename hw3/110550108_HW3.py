import os
import random
import json
import tempfile
import gc
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from PIL import Image
import cv2
import matplotlib.pyplot as plt


# Configuration
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = 'vr-hw3/data'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
TEST_DIR = os.path.join(DATA_ROOT, 'test_release')
TEST_INFO_PATH = os.path.join(DATA_ROOT, 'test_image_name_to_ids.json')
SAVE_DIR = 'output'
RESULT_PATH = 'output/test-results.json'
NUM_CLASSES = 5  # Background + 4 classes
VAL_RATIO = 0.2
SEED = 42


if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def set_random_seeds():
    """Set random seeds for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def binary_mask_to_rle(binary_mask):
    """Convert a binary mask to RLE format."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def get_bbox_from_mask(mask):
    """Get bounding box from mask."""
    pos = np.where(mask > 0)
    if len(pos[0]) == 0:  # Empty mask
        return [0, 0, 1, 1]
    
    xmin = np.min(pos[1])
    ymin = np.min(pos[0])
    xmax = np.max(pos[1])
    ymax = np.max(pos[0])
    if xmax > xmin and ymax > ymin:
        return [int(xmin), int(ymin), int(xmax), int(ymax)]
    else:
        return [0, 0, 1, 1]


class CellSegmentationDataset(Dataset):
    """Custom dataset class for cell segmentation."""
    
    def __init__(self, data_dirs, transform=None):
        self.data_dirs = data_dirs
        self.transform = transform
        self.samples = []
        
        for data_dir in self.data_dirs:
            image_path = os.path.join(data_dir, 'image.tif')
            if not os.path.exists(image_path):
                continue
            
            # Find class masks
            masks = []
            for class_id in range(1, NUM_CLASSES):  # 1-4 for classes
                class_name = f'class{class_id}'
                class_mask_path = os.path.join(data_dir, f'{class_name}.tif')
                if os.path.exists(class_mask_path):
                    masks.append((class_id, class_mask_path))
            
            # Only add samples with at least one mask
            if masks:
                self.samples.append((image_path, masks))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, masks_info = self.samples[idx]
        image = np.array(Image.open(image_path))
        
        # Handle RGBA images
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Use only RGB channels
        
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        # Process masks
        mask_tensors = []
        boxes = []
        labels = []
        
        for class_id, mask_path in masks_info:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]  # Remove background
            
            for instance_id in instance_ids:
                instance_mask = (mask == instance_id).astype(np.uint8)
                bbox = get_bbox_from_mask(instance_mask)
                mask_tensors.append(torch.as_tensor(instance_mask, dtype=torch.uint8))
                boxes.append(bbox)
                labels.append(class_id)
        
        # Create target dictionary
        target = {}
        if mask_tensors:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            target["masks"] = torch.stack(mask_tensors) if mask_tensors else torch.zeros(
                (0, mask.shape[0], mask.shape[1]), dtype=torch.uint8
            )
        else:
            # Empty sample
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = torch.zeros(
                (0, image.shape[1], image.shape[2]), dtype=torch.uint8
            )
        
        del masks_info, labels, boxes
        return image, target


def collate_fn(batch):
    """Collate function to handle different sized images in a batch."""
    return tuple(zip(*batch))


def get_model(num_classes=NUM_CLASSES):
    """Create and configure Mask R-CNN model."""
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_detections_per_img=200
    )
    
    model.transform.min_size = [640]
    model.transform.max_size = 1066
    
    # Modify anchor sizes and aspect ratios for cell detection
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 1.5, 2.0),) * len(anchor_sizes)
    model.rpn.anchor_generator.sizes = anchor_sizes
    model.rpn.anchor_generator.aspect_ratios = aspect_ratios
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 1024, num_classes
    )
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    # Training progress bar
    with tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision
            with autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # Scale loss and backpropagate
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update progress bar
            total_loss += losses.item()
            pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")
            
            # Free memory
            del images, targets, loss_dict, losses
            torch.cuda.empty_cache()
    
    return total_loss / len(data_loader)


def validate(model, data_loader, device):
    """Validate the model on validation data."""
    model.eval()
    results = []
    
    # Prepare COCO format ground truth
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for i in range(1, NUM_CLASSES):
        coco_gt["categories"].append({
            "id": i,
            "name": f"class{i}",
            "supercategory": "cell"
        })
    
    image_id = 0
    ann_id = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                img_id = image_id
                image_id += 1
                
                image_height, image_width = images[i].shape[1], images[i].shape[2]
                coco_gt["images"].append({
                    "id": img_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": f"val_image_{img_id}.jpg"
                })
                
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                gt_masks = target["masks"].cpu().numpy()
                
                for box_idx in range(len(gt_boxes)):
                    mask_rle = binary_mask_to_rle(gt_masks[box_idx])
                    
                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(gt_labels[box_idx]),
                        "bbox": gt_boxes[box_idx].tolist(),
                        "area": float(mask_utils.area(mask_rle)),
                        "segmentation": mask_rle,
                        "iscrowd": 0
                    })
                    ann_id += 1
                
                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_masks = output["masks"].cpu().numpy()
                
                for box_idx in range(len(pred_boxes)):
                    # Convert mask to RLE format (threshold the mask)
                    mask = (pred_masks[box_idx, 0] > 0.5).astype(np.uint8)
                    mask_rle = binary_mask_to_rle(mask)
                    
                    results.append({
                        "image_id": img_id,
                        "category_id": int(pred_labels[box_idx]),
                        "bbox": pred_boxes[box_idx].tolist(),
                        "segmentation": mask_rle,
                        "score": float(pred_scores[box_idx])
                    })
            
            # Free memory
            del images, targets, outputs, gt_masks, gt_labels
            del pred_masks, pred_labels, pred_scores
            torch.cuda.empty_cache()
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as pred_f, \
         tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as gt_f:

        json.dump(results, pred_f)
        json.dump(coco_gt, gt_f)
        pred_f.flush()
        gt_f.flush()

        try:
            coco_gt_obj = COCO(gt_f.name)
            coco_dt = coco_gt_obj.loadRes(pred_f.name)

            coco_eval = COCOeval(coco_gt_obj, coco_dt, iouType='segm')
            coco_eval.params.iouThrs = np.array([0.5], dtype=float)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            map_score = coco_eval.stats[0]
        except Exception as e:
            print(f"Error in COCO evaluation: {e}")
            map_score = 0.0
    
    return map_score, results


def load_and_split_data():
    """Load and split data into training and validation sets."""
    all_dirs = [
        os.path.join(TRAIN_DIR, d) for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ]
    
    # Split into train and validation
    num_val = int(len(all_dirs) * VAL_RATIO)
    random.shuffle(all_dirs)
    train_dirs = all_dirs[num_val:]
    val_dirs = all_dirs[:num_val]
    
    print(f"Training samples: {len(train_dirs)}, Validation samples: {len(val_dirs)}")
    
    # Create datasets
    train_dataset = CellSegmentationDataset(train_dirs)
    val_dataset = CellSegmentationDataset(val_dirs)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def run_test(model, device):
    """Run inference on test data."""
    model.eval()
    results = []
    
    with open(TEST_INFO_PATH, 'r') as f:
        test_info = json.load(f)
    
    # Create a mapping from filename to image id
    filename_to_id = {item['file_name']: item['id'] for item in test_info}
    
    for test_file in tqdm(os.listdir(TEST_DIR), desc="Testing"):
        if not test_file.endswith('.tif'):
            continue
        
        image_id = filename_to_id.get(test_file)
        if image_id is None:
            print(f"Warning: No ID found for {test_file}")
            continue
        
        image_path = os.path.join(TEST_DIR, test_file)
        image = np.array(Image.open(image_path))
        
        # Handle RGBA images
        if image.ndim == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Use only RGB channels
        
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model([image_tensor])[0]
        
        pred_boxes = output["boxes"].cpu().numpy()
        pred_labels = output["labels"].cpu().numpy()
        pred_scores = output["scores"].cpu().numpy()
        pred_masks = output["masks"].cpu().numpy()
        
        for box_idx in range(len(pred_boxes)):
            # Filter by score threshold
            if pred_scores[box_idx] < 0.8:
                continue
                
            mask = (pred_masks[box_idx, 0] > 0.5).astype(np.uint8)
            mask_rle = binary_mask_to_rle(mask)
            
            results.append({
                "image_id": int(image_id),
                "category_id": int(pred_labels[box_idx]),
                "bbox": pred_boxes[box_idx].tolist(),
                "segmentation": mask_rle,
                "score": float(pred_scores[box_idx])
            })
        # Clear memory after each image
        del image, image_tensor, output
        torch.cuda.empty_cache()
    
    with open(RESULT_PATH, 'w') as f:
        json.dump(results, f)
    
    print(f"Saved {len(results)} predictions to {RESULT_PATH}")
    return results


def plot_learning_curve(epochs_list, loss_values, map_values):
    """Plot and save learning curves."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, loss_values, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, map_values, 'r-', label='Validation mAP')
    plt.title('Validation mAP')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'learning_curve.png'))
    plt.close()


def main():
    """Main function to run training and evaluation."""
    # Set random seeds
    set_random_seeds()
    
    # Load and split data
    train_loader, val_loader = load_and_split_data()
    
    # Create model
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # Calculate model size
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {model_parameters / 1e6:.2f}M")
    
    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE/100)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_map = 0.0
    best_model_path = os.path.join(SAVE_DIR, 'best_model.pth')
    
    train_losses = []
    val_maps = []
    epochs = []
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, scaler)
        train_losses.append(train_loss)
        torch.cuda.empty_cache()
        
        # Validate
        val_map, _ = validate(model, val_loader, DEVICE)
        val_maps.append(val_map)
        epochs.append(epoch + 1)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, "
              f"Val mAP@0.5: {val_map:.4f}")
        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with mAP: {best_map:.4f}")
        
        # Plot learning curve
        plot_learning_curve(epochs, train_losses, val_maps)
        
        # Free GPU memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    
    # Run test
    test_results = run_test(model, DEVICE)
    
    # Final plot
    plot_learning_curve(epochs, train_losses, val_maps)
    
    print("Training and evaluation complete!")
    print(f"Best validation mAP@0.5: {best_map:.4f}")
    print(f"Results saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()