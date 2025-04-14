import os
import json
import csv

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from tqdm import tqdm
from torch.amp import autocast, GradScaler


NUM_EPOCHS = 5
BATCH_SIZE = 32
ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4

output_dir = './output/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


class DigitDataset(torch.utils.data.Dataset):
    """Dataset for digit detection using COCO format annotations."""
    
    def __init__(self, root, json_file, transform):
        self.root = root
        self.transform = transform
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = data['categories']
        
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        target = {}
        if img_id in self.img_to_anns:
            boxes = []
            labels = []
            areas = []
            
            for ann in self.img_to_anns[img_id]:
                # COCO format: [x_min, y_min, width, height]
                # Convert to [x_min, y_min, x_max, y_max]
                bbox = ann['bbox']
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])
                areas.append(ann['area'])
            
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.as_tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Empty annotations
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.zeros(0, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(0, dtype=torch.int64)
        
        img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.images)


class TestDigitDataset(torch.utils.data.Dataset):
    """Dataset for digit detection test set without annotations."""
    
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if f.endswith('.png')]
        self.image_files.sort(key=lambda x: int(x.split('.')[0]))
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_id = int(img_file.split('.')[0])
        img_path = os.path.join(self.root, img_file)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        return img, img_id
    
    def __len__(self):
        return len(self.image_files)


def get_model(num_classes):
    """Create a Faster R-CNN model."""
    
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=5
    )
    
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.25, 0.5, 1.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator
    model.roi_heads.detections_per_img = 10
    model.transform.min_size = [128]
    model.transform.max_size = 1024
    model.rpn.pre_nms_top_n_train = 1000
    model.rpn.pre_nms_top_n_test = 500
    model.rpn.post_nms_top_n_train = 500
    model.rpn.post_nms_top_n_test = 250
    model.rpn.nms_thresh = 0.6
    model.roi_heads.nms_thresh = 0.4
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, scaler,
                    accumulation_steps, scheduler):
    """Train the model for one epoch."""
    
    model.train()
    
    running_loss = 0.0
    steps = 0
    optimizer.zero_grad()
    
    for i, (images, targets) in enumerate(tqdm(data_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with autocast(device_type='cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        loss_value = losses.item()
        normalized_loss = losses / accumulation_steps
        scaler.scale(normalized_loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        running_loss += loss_value
        steps += 1
    
    # Handle any remaining gradients at the end of the epoch
    if (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    
    return running_loss / steps


def evaluate(model, data_loader, device):
    """Evaluate the model on validation data."""
    
    model.train()  # Only train mode will return loss
    
    total_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                      for t in targets]
            
            with autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            steps += 1
    
    return total_loss / steps


def predict_digits(model, data_loader, device):
    """Generate predictions for test images."""
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, image_ids in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            
            with autocast(device_type='cuda'):
                outputs = model(images)
            
            for i, output in enumerate(outputs):
                if isinstance(image_ids[i], torch.Tensor):
                    image_id = image_ids[i].item()
                else:
                    image_id = image_ids[i]
                
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Convert back to COCO format [x_min, y_min, width, height]
                for j in range(len(boxes)):
                    x_min, y_min, x_max, y_max = boxes[j]
                    width = x_max - x_min
                    height = y_max - y_min
                    box = [
                        float(x_min), float(y_min),
                        float(width), float(height)
                    ]
                    
                    predictions.append({
                        'image_id': image_id,
                        'bbox': box,
                        'score': float(scores[j]),
                        'category_id': int(labels[j])
                    })
    
    return predictions


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Add small epsilon to prevent division by zero
    return intersection / (union + 1e-6)


def recognize_numbers(predictions, test_dataset):
    """Recognize multi-digit numbers from individual digit predictions."""
    
    predictions_by_image = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in predictions_by_image:
            predictions_by_image[img_id] = []
        predictions_by_image[img_id].append(pred)
    
    results = []
    for img_id in range(1, len(test_dataset) + 1):
        if img_id not in predictions_by_image or not predictions_by_image[img_id]:
            results.append([img_id, -1])
            continue
        
        preds = [p for p in predictions_by_image[img_id] if p['score'] > 0.7]
        
        if not preds:
            results.append([img_id, -1])
            continue
        
        boxes = np.array([p['bbox'] for p in preds])
        scores = np.array([p['score'] for p in preds])
        
        # Convert [x, y, w, h] to [x1, y1, x2, y2] for IoU calculation
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0]
        boxes_xyxy[:, 1] = boxes[:, 1]
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # Apply NMS to remove overlapping detections
        indices = np.argsort(-scores)
        keep_indices = []
        
        while len(indices) > 0:
            current = indices[0]
            keep_indices.append(current)
            indices = indices[1:]
            
            if len(indices) == 0:
                break
            
            current_box = boxes_xyxy[current]
            ious = np.array([
                calculate_iou(current_box, boxes_xyxy[i]) for i in indices
            ])
            indices = indices[ious < 0.3]
        
        filtered_preds = [preds[i] for i in keep_indices]
        filtered_preds.sort(key=lambda p: p['bbox'][0])
        
        digits = []
        for p in filtered_preds:
            digit = p['category_id'] - 1  # category_id starts from 1
            digits.append(str(digit))
        
        number = int(''.join(digits)) if digits else -1
        results.append([img_id, number])
    
    return results


def visualize_predictions(model, dataset, device, num_samples=5):
    """Visualize model predictions and ground truth on sample images."""
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    model.eval()
    _, axs = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    
    sample_count = 0
    with torch.no_grad():
        for images, targets in dataloader:
            if sample_count >= num_samples:
                break
            
            image = images[0].clone()
            target = targets[0]
            
            img_np = image.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            axs[sample_count, 0].imshow(img_np)
            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy()
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor='g', linewidth=2
                )
                axs[sample_count, 0].add_patch(rect)
                axs[sample_count, 0].text(
                    x1, y1, str(label - 1),
                    bbox=dict(facecolor='white', alpha=0.7)
                )
            axs[sample_count, 0].set_title('Ground Truth')
            axs[sample_count, 0].axis('off')
            
            image = image.to(device)
            with autocast(device_type='cuda'):
                output = model([image])[0]
            
            axs[sample_count, 1].imshow(img_np)
            pred_boxes = output['boxes'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            
            keep = pred_scores > 0.7
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor='r', linewidth=2
                )
                axs[sample_count, 1].add_patch(rect)
                axs[sample_count, 1].text(
                    x1, y1, f"{label - 1} ({score:.2f})",
                    bbox=dict(facecolor='white', alpha=0.7)
                )
            axs[sample_count, 1].set_title('Predictions')
            axs[sample_count, 1].axis('off')
            
            sample_count += 1
    
    plt.tight_layout()
    plt.savefig('output/predictions_visualization.png')
    plt.close()


def main():
        
    # Paths
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    train_json = os.path.join(data_dir, 'train.json')
    valid_json = os.path.join(data_dir, 'valid.json')
    
    # Transforms
    transform = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DigitDataset(train_dir, train_json, transform=transform)
    valid_dataset = DigitDataset(valid_dir, valid_json, transform=transform)
    test_dataset = TestDigitDataset(test_dir, transform=transform)
    
    # Data loaders
    batch_size = BATCH_SIZE
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: tuple(zip(*batch)),
        pin_memory=True,
        persistent_workers=True,
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: tuple(zip(*batch)),
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: tuple(zip(*batch)),
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=11)  # 10 digits + background
    model.to(device)
    
    # Gradient accumulation steps
    accumulation_steps = ACCUMULATION_STEPS
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    num_epochs = NUM_EPOCHS
    steps_per_epoch = len(train_loader) // accumulation_steps
    
    if len(train_loader) % accumulation_steps != 0:
        steps_per_epoch += 1
    
    total_steps = steps_per_epoch * num_epochs
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=10,
        final_div_factor=1000
    )
    
    best_model_path = 'output/best_model.pth'
    best_loss = float('inf')
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, scaler,
            accumulation_steps, scheduler
        )
        val_loss = evaluate(model, valid_loader, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('output/loss_curve.png')
    plt.close()
    
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    visualize_predictions(model, valid_dataset, device)
    test_predictions = predict_digits(model, test_loader, device)
    
    # Save Task 1 results (detection)
    with open('output/pred.json', 'w') as f:
        json.dump(test_predictions, f)
    
    # Recognize numbers from detected digits
    number_predictions = recognize_numbers(test_predictions, test_dataset)
    
    # Save Task 2 results (recognition)
    with open('output/pred.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'pred_label'])
        writer.writerows(number_predictions)
    
    print("Predictions saved to pred.json and pred.csv")


if __name__ == "__main__":
    main()
