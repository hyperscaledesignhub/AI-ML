import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# DATASET: Street View Images with Face/License Plate Labels
# ============================================================================

class StreetViewDataset(Dataset):
    """
    Dataset containing Street View images with ground truth labels
    WHY: Model needs to learn from labeled examples
    """
    
    def __init__(self, image_paths, annotations):
        """
        Args:
            image_paths: List of image file paths
            annotations: List of dicts with 'boxes' and 'labels'
        """
        self.image_paths = image_paths
        self.annotations = annotations
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns one training example
        """
        # Load image (simplified - normally use PIL/OpenCV)
        image = torch.rand(3, 800, 600)  # Placeholder
        
        # Get ground truth annotations
        # Format: {'boxes': [[x1,y1,x2,y2], ...], 'labels': [1, 2, 1, ...]}
        # Labels: 0=background, 1=face, 2=license_plate
        target = self.annotations[idx]
        
        return image, target


# ============================================================================
# LOSS FUNCTIONS
# WHY: Need to measure how wrong our predictions are
# ============================================================================

def calculate_rpn_loss(rpn_objectness, rpn_boxes, gt_boxes):
    """
    RPN Loss = How well did RPN propose regions?
    
    Two components:
    1. Objectness loss: Did we correctly identify foreground vs background?
    2. Box regression loss: Are proposed boxes close to ground truth?
    
    WHY: RPN needs to learn which areas contain objects
    """
    
    # ----------------------------------------------------------------
    # Part 1: Objectness Loss (Classification)
    # WHY: Teach RPN to distinguish "something here" vs "nothing here"
    # ----------------------------------------------------------------
    
    # For each anchor, determine if it overlaps with ground truth
    # If overlap > 0.7 → positive (label=1, "foreground")
    # If overlap < 0.3 → negative (label=0, "background")
    # Else → ignore (don't train on these)
    
    positive_anchors, negative_anchors = match_anchors_to_gt(rpn_boxes, gt_boxes)
    
    # Binary cross-entropy loss
    objectness_loss = nn.CrossEntropyLoss()
    
    # Calculate loss only on positive and negative anchors
    loss_obj = objectness_loss(rpn_objectness, positive_anchors + negative_anchors)
    
    # WHY cross-entropy? Standard for classification problems
    
    # ----------------------------------------------------------------
    # Part 2: Box Regression Loss
    # WHY: Teach RPN to adjust anchor boxes to fit objects
    # ----------------------------------------------------------------
    
    # Only calculate box loss on positive anchors
    # (no point adjusting boxes where there's no object)
    
    # Smooth L1 loss: less sensitive to outliers than L2
    box_loss_fn = nn.SmoothL1Loss()
    loss_box = box_loss_fn(rpn_boxes[positive_anchors], gt_boxes[positive_anchors])
    
    # WHY Smooth L1? 
    # - Robust to outliers (some boxes might be way off)
    # - Gradients don't explode for large errors
    
    # Total RPN loss
    rpn_loss = loss_obj + loss_box
    
    return rpn_loss


def calculate_roi_loss(roi_classes, roi_boxes, gt_classes, gt_boxes):
    """
    ROI Head Loss = How well did we classify and localize objects?
    
    Two components:
    1. Classification loss: Did we predict correct class (face/plate)?
    2. Box regression loss: Are final boxes accurate?
    
    WHY: ROI Head needs to learn what each object is and where exactly
    """
    
    # ----------------------------------------------------------------
    # Part 1: Classification Loss
    # WHY: Teach model to recognize faces vs license plates
    # ----------------------------------------------------------------
    
    # Cross-entropy loss for multi-class classification
    cls_loss_fn = nn.CrossEntropyLoss()
    loss_cls = cls_loss_fn(roi_classes, gt_classes)
    
    # roi_classes: [2000, 3] - predicted probabilities
    # gt_classes: [2000] - true labels (0, 1, or 2)
    
    # WHY cross-entropy? Standard for multi-class problems
    
    # ----------------------------------------------------------------
    # Part 2: Box Regression Loss
    # WHY: Teach model precise object localization
    # ----------------------------------------------------------------
    
    # Only calculate box loss for foreground classes (not background)
    foreground_mask = gt_classes > 0
    
    box_loss_fn = nn.SmoothL1Loss()
    loss_box = box_loss_fn(
        roi_boxes[foreground_mask], 
        gt_boxes[foreground_mask]
    )
    
    # WHY only foreground? Background has no box to refine
    
    # Total ROI loss
    roi_loss = loss_cls + loss_box
    
    return roi_loss


def match_anchors_to_gt(anchors, gt_boxes, pos_threshold=0.7, neg_threshold=0.3):
    """
    Assign each anchor as positive, negative, or ignore
    
    WHY: Need to know which anchors should detect objects vs background
    
    Args:
        anchors: Predicted anchor boxes
        gt_boxes: Ground truth boxes
        pos_threshold: IoU > 0.7 → positive
        neg_threshold: IoU < 0.3 → negative
    """
    # Calculate IoU (Intersection over Union) between anchors and GT
    ious = calculate_iou(anchors, gt_boxes)
    
    # Assign labels
    positive = ious > pos_threshold  # These should detect objects
    negative = ious < neg_threshold  # These should detect background
    
    return positive, negative


def calculate_iou(boxes1, boxes2):
    """
    Calculate Intersection over Union
    
    WHY: Measure how much two boxes overlap
    - IoU = 1.0: Perfect overlap
    - IoU = 0.0: No overlap
    - IoU = 0.5: Half overlapping
    """
    # Intersection area
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection
    
    iou = intersection / union
    return iou


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_faster_rcnn(model, train_dataset, num_epochs=50, learning_rate=0.001):
    """
    Complete training process
    
    WHY: Teach model to detect faces and license plates accurately
    """
    
    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    
    # DataLoader: Batch images together
    # WHY batch_size=2? Faster R-CNN uses large images, limited GPU memory
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,      # Small batch due to large images
        shuffle=True,      # WHY shuffle? Prevent learning order patterns
        num_workers=4      # WHY 4? Parallel data loading for speed
    )
    
    # Optimizer: Update model weights
    # WHY Adam? Good default, adaptive learning rates
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    # WHY? Reduce learning rate over time for fine-tuning
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10,      # Every 10 epochs
        gamma=0.1          # Reduce LR by 10x
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Training on: {device}")
    print(f"Total epochs: {num_epochs}")
    print(f"Training samples: {len(train_dataset)}")
    
    # ----------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------
    
    for epoch in range(num_epochs):
        
        model.train()  # WHY? Enable dropout, batch norm training mode
        
        epoch_loss = 0
        epoch_rpn_loss = 0
        epoch_roi_loss = 0
        
        # ----------------------------------------------------------------
        # Iterate over batches
        # ----------------------------------------------------------------
        for batch_idx, (images, targets) in enumerate(train_loader):
            
            # Move data to GPU
            images = images.to(device)
            # targets: List of dicts with 'boxes' and 'labels'
            
            # ----------------------------------------------------------------
            # FORWARD PASS: Get predictions
            # ----------------------------------------------------------------
            
            # Step 1: Extract features
            feature_maps = model.backbone(images)
            
            # Step 2: RPN proposals
            rpn_objectness, rpn_boxes = model.rpn(feature_maps)
            
            # Step 3: ROI Head predictions
            roi_classes, roi_boxes = model.roi_head(feature_maps, rpn_boxes)
            
            # ----------------------------------------------------------------
            # CALCULATE LOSSES
            # WHY separate losses? Different parts learn different tasks
            # ----------------------------------------------------------------
            
            # RPN Loss: How well did we propose regions?
            rpn_loss = calculate_rpn_loss(
                rpn_objectness, 
                rpn_boxes, 
                targets  # Ground truth boxes
            )
            
            # ROI Loss: How well did we classify and refine?
            roi_loss = calculate_roi_loss(
                roi_classes,
                roi_boxes,
                targets  # Ground truth labels and boxes
            )
            
            # Total loss
            total_loss = rpn_loss + roi_loss
            
            # WHY add them? Both stages must learn together
            
            # ----------------------------------------------------------------
            # BACKWARD PASS: Update weights
            # ----------------------------------------------------------------
            
            # Clear previous gradients
            optimizer.zero_grad()
            # WHY? Gradients accumulate by default in PyTorch
            
            # Compute gradients
            total_loss.backward()
            # WHY? Calculate how to adjust each weight
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # WHY? Object detection models can have unstable gradients
            
            # Update weights
            optimizer.step()
            # WHY? Actually adjust the model parameters
            
            # ----------------------------------------------------------------
            # Track progress
            # ----------------------------------------------------------------
            epoch_loss += total_loss.item()
            epoch_rpn_loss += rpn_loss.item()
            epoch_roi_loss += roi_loss.item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"(RPN: {rpn_loss.item():.4f}, "
                      f"ROI: {roi_loss.item():.4f})")
        
        # ----------------------------------------------------------------
        # End of epoch
        # ----------------------------------------------------------------
        
        avg_loss = epoch_loss / len(train_loader)
        avg_rpn = epoch_rpn_loss / len(train_loader)
        avg_roi = epoch_roi_loss / len(train_loader)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Average Total Loss: {avg_loss:.4f}")
        print(f"  Average RPN Loss:   {avg_rpn:.4f}")
        print(f"  Average ROI Loss:   {avg_roi:.4f}")
        print(f"{'='*60}\n")
        
        # Update learning rate
        scheduler.step()
        # WHY? Smaller learning rate over time = finer adjustments
        
        # ----------------------------------------------------------------
        # Save checkpoint every 5 epochs
        # WHY? Don't lose progress if training crashes
        # ----------------------------------------------------------------
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'faster_rcnn_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}\n")
    
    # ----------------------------------------------------------------
    # Save final model
    # ----------------------------------------------------------------
    torch.save(model.state_dict(), 'faster_rcnn_final.pth')
    print("✅ Training complete! Final model saved.")
    
    return model


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # ----------------------------------------------------------------
    # Prepare dataset (simplified example)
    # ----------------------------------------------------------------
    print("Preparing dataset...")
    
    # Example: 1000 Street View images with annotations
    image_paths = [f'streetview_{i}.jpg' for i in range(1000)]
    
    # Example annotations
    annotations = [
        {
            'boxes': torch.tensor([[100, 150, 200, 250],   # Face
                                   [400, 300, 550, 380]]), # License plate
            'labels': torch.tensor([1, 2])  # 1=face, 2=plate
        }
        for _ in range(1000)
    ]
    
    dataset = StreetViewDataset(image_paths, annotations)
    
    # ----------------------------------------------------------------
    # Initialize model
    # ----------------------------------------------------------------
    print("Initializing model...")
    model = FasterRCNN(num_classes=3)  # background, face, license_plate
    
    # ----------------------------------------------------------------
    # Train model
    # ----------------------------------------------------------------
    print("Starting training...\n")
    
    trained_model = train_faster_rcnn(
        model=model,
        train_dataset=dataset,
        num_epochs=50,
        learning_rate=0.001
    )
    
    # ----------------------------------------------------------------
    # Use trained model for inference
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("Training complete! Now you can use the model:")
    print("="*60)
    
    # Load trained weights
    trained_model.load_state_dict(torch.load('faster_rcnn_final.pth'))
    trained_model.eval()
    
    # Test on new image
    test_image = torch.rand(1, 3, 800, 600)
    with torch.no_grad():
        detections = detect_faces_and_plates(trained_model, test_image)
    
    print(f"Detected {len(detections)} faces/plates in test image")
```

---

## 📊 Training Progress Visualization
```
Epoch [1/50] Batch [0/500] Loss: 2.5432 (RPN: 1.2341, ROI: 1.3091)
Epoch [1/50] Batch [100/500] Loss: 1.8765 (RPN: 0.9234, ROI: 0.9531)
Epoch [1/50] Batch [200/500] Loss: 1.3456 (RPN: 0.6543, ROI: 0.6913)
...
============================================================
Epoch 1/50 Summary:
  Average Total Loss: 1.4523
  Average RPN Loss:   0.7123
  Average ROI Loss:   0.7400
============================================================

✅ Checkpoint saved: faster_rcnn_epoch_5.pth
...
✅ Training complete! Final model saved.
```

---

## 🎯 Key Training Concepts

### 1. **Why Two Losses?**
```
RPN Loss → Teaches: "Find candidate regions"
ROI Loss → Teaches: "Classify those regions"

Both must learn together!
```

### 2. **Why Small Batch Size (2)?**
```
Faster R-CNN:
- Large images (800×600)
- 2000 proposals per image
- Lots of GPU memory needed

Batch=2 fits in typical GPU (8-12GB)
```

### 3. **Why Learning Rate Decay?**
```
Early training: Large steps (lr=0.001)
Later training: Small steps (lr=0.0001)

Like: Rough sketch → Fine details