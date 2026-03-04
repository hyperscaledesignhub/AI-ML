import torch
import torch.nn as nn
import torchvision.models as models

# ============================================================================
# COMPLETE FASTER R-CNN FOR STREET VIEW BLURRING
# ============================================================================

class FasterRCNN(nn.Module):
    """
    Two-Stage Object Detector:
    - Stage 1: Find WHERE objects might be (RPN)
    - Stage 2: Find WHAT those objects are (ROI Head)
    """
    
    def __init__(self, num_classes=3):  # 3 = background, face, license_plate
        super(FasterRCNN, self).__init__()
        
        # ----------------------------------------------------------------
        # BACKBONE: Extract features from raw images
        # WHY: Raw pixels are too low-level. We need high-level features
        #      like "edges", "textures", "shapes" that help detect objects
        # ----------------------------------------------------------------
        resnet = models.resnet50(pretrained=True)  # Pre-trained on ImageNet
        # WHY pretrained? Starting from scratch would need 10M+ images
        # ResNet already knows basic features (edges, corners, textures)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # WHY remove last 2 layers? Those are for ImageNet classification
        # We only need feature extraction, not classification
        
        self.feature_dim = 2048  # ResNet-50 outputs 2048 channels
        
        # ----------------------------------------------------------------
        # STAGE 1: Region Proposal Network (RPN)
        # WHY: Can't check every possible box in image (millions of them!)
        #      RPN narrows down to ~2000 promising locations
        # ----------------------------------------------------------------
        self.rpn = RegionProposalNetwork(self.feature_dim)
        
        # ----------------------------------------------------------------
        # STAGE 2: ROI Head (Classifier + Box Refiner)
        # WHY: RPN only says "something is here"
        #      ROI Head says "it's a FACE" or "it's a LICENSE PLATE"
        # ----------------------------------------------------------------
        self.roi_head = ROIHead(self.feature_dim, num_classes)
        
    def forward(self, images):
        """
        Complete forward pass through both stages
        
        Args:
            images: [batch, 3, 800, 600] - RGB images from Street View
            
        Returns:
            classes: What each detected object is
            boxes: Where each detected object is
        """
        # ----------------------------------------------------------------
        # STEP 1: Extract Features
        # WHY: Convert raw pixels to meaningful features
        # ----------------------------------------------------------------
        feature_maps = self.backbone(images)  
        # Input:  [1, 3, 800, 600]     - RGB image
        # Output: [1, 2048, 50, 38]    - Feature map (downsampled 16x)
        # WHY 50×38? ResNet uses pooling/stride, reduces 800→50, 600→38
        
        # ----------------------------------------------------------------
        # STEP 2: STAGE 1 - Propose Regions (RPN)
        # WHY: Find candidate locations before expensive classification
        # ----------------------------------------------------------------
        proposals, proposal_scores = self.rpn(feature_maps)
        # Input:  Feature maps [1, 2048, 50, 38]
        # Output: Top 2000 boxes that might contain objects
        # WHY 2000? Balance between recall (find all objects) and speed
        
        # ----------------------------------------------------------------
        # STEP 3: STAGE 2 - Classify and Refine (ROI Head)
        # WHY: Know WHAT each proposed region contains
        # ----------------------------------------------------------------
        classes, refined_boxes = self.roi_head(feature_maps, proposals)
        # Input:  Feature maps + 2000 proposals
        # Output: Class (face/plate/background) + precise box coordinates
        
        return classes, refined_boxes


# ============================================================================
# STAGE 1: REGION PROPOSAL NETWORK (RPN)
# PURPOSE: Quickly scan image and find ~2000 candidate object locations
# ============================================================================

class RegionProposalNetwork(nn.Module):
    """
    Scans feature map and proposes: "I think something is HERE"
    Does NOT know WHAT it is (that's Stage 2's job)
    """
    
    def __init__(self, in_channels=2048):
        super(RegionProposalNetwork, self).__init__()
        
        # ----------------------------------------------------------------
        # Define Anchor Templates
        # WHY: Objects come in different sizes and shapes
        #      Faces: small, square
        #      License plates: medium, rectangular
        # ----------------------------------------------------------------
        self.anchor_scales = [64, 128, 256]       # Small, medium, large
        self.anchor_ratios = [0.5, 1.0, 2.0]      # Tall, square, wide
        self.num_anchors = 3 * 3  # = 9 anchor templates per position
        
        # WHY 9? Cover different object sizes/shapes without checking millions
        
        # ----------------------------------------------------------------
        # Sliding Window Scanner
        # WHY: Need to check every location in the image for objects
        # ----------------------------------------------------------------
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        # WHY kernel_size=3? 
        # - 3×3 window slides across feature map
        # - At each position, looks at 3×3 neighborhood (48×48 pixels in original image)
        # - Small enough to be fast, large enough to see context
        
        # ----------------------------------------------------------------
        # TWO PREDICTIONS at each position:
        # ----------------------------------------------------------------
        
        # Prediction 1: "Is there an object here?" (Objectness)
        # WHY: Most locations are empty (background)
        #      This filters out 95% of useless locations quickly
        self.objectness = nn.Conv2d(512, self.num_anchors * 2, kernel_size=1)
        # Output: 9 anchors × 2 scores = 18 values per position
        # 2 scores = [background_score, foreground_score]
        
        # Prediction 2: "How to adjust the anchor box to fit better?"
        # WHY: Anchor templates are rough guesses
        #      Need fine adjustments to fit actual object tightly
        self.box_regression = nn.Conv2d(512, self.num_anchors * 4, kernel_size=1)
        # Output: 9 anchors × 4 adjustments = 36 values per position
        # 4 adjustments = [Δx, Δy, Δwidth, Δheight]
        
    def forward(self, feature_maps):
        """
        Scan feature map and propose candidate regions
        
        Args:
            feature_maps: [1, 2048, 50, 38] from backbone
            
        Returns:
            proposals: Top 2000 boxes [2000, 4] as [x1, y1, x2, y2]
            scores: Confidence for each proposal
        """
        batch_size = feature_maps.size(0)
        
        # ----------------------------------------------------------------
        # Slide 3×3 window across entire feature map
        # WHY: Check every possible location for objects
        # ----------------------------------------------------------------
        x = torch.relu(self.conv(feature_maps))  
        # Input:  [1, 2048, 50, 38]
        # Output: [1, 512, 50, 38]
        # Still 50×38 positions, but with 512 features per position
        
        # ----------------------------------------------------------------
        # At each of 50×38 positions, make predictions for 9 anchors
        # ----------------------------------------------------------------
        
        # Predict: "Is something here?"
        objectness_scores = self.objectness(x)  
        # Output: [1, 18, 50, 38]
        # 18 = 9 anchors × 2 scores (background/foreground)
        # WHY? Need to know which of the 9 anchor shapes fits best
        
        # Predict: "How to adjust each anchor?"
        box_deltas = self.box_regression(x)  
        # Output: [1, 36, 50, 38]
        # 36 = 9 anchors × 4 coordinates (x, y, w, h adjustments)
        # WHY? Anchors are templates; real objects need fine-tuning
        
        # ----------------------------------------------------------------
        # Generate actual proposal boxes
        # WHY: Convert predictions into actual coordinate values
        # ----------------------------------------------------------------
        all_proposals = self._generate_proposals(objectness_scores, box_deltas)
        # Total: 50 × 38 × 9 = 17,100 candidate boxes!
        
        # ----------------------------------------------------------------
        # Keep only top 2000 proposals
        # WHY: Can't process all 17,100 in Stage 2 - too slow
        #      2000 is enough to catch all faces/plates (high recall)
        # ----------------------------------------------------------------
        top_proposals = self._filter_top_2000(all_proposals, objectness_scores)
        
        return top_proposals, objectness_scores
    
    def _generate_proposals(self, objectness_scores, box_deltas):
        """
        Convert anchor templates + deltas into actual box coordinates
        WHY: Need concrete [x, y, w, h] values for each proposal
        """
        # Implementation details...
        # 1. Generate base anchor boxes at each feature map position
        # 2. Apply predicted deltas to adjust anchors
        # 3. Clip boxes to image boundaries
        proposals = torch.rand(1, 17100, 4)  # Simplified placeholder
        return proposals
    
    def _filter_top_2000(self, proposals, scores):
        """
        Keep top 2000 boxes by objectness score
        WHY: Balance between finding all objects and computational cost
        """
        # Get foreground scores (confident there's an object)
        fg_scores = scores[:, 1::2, :, :].flatten()  # Every 2nd channel
        
        # Sort and keep top 2000
        top_indices = torch.topk(fg_scores, k=2000)[1]
        top_proposals = proposals[:, top_indices]
        
        return top_proposals


# ============================================================================
# STAGE 2: ROI HEAD (Region of Interest Head)
# PURPOSE: For each proposal from RPN, determine WHAT it is
# ============================================================================

class ROIHead(nn.Module):
    """
    Takes RPN proposals and answers:
    1. WHAT is this? (face, license plate, or background)
    2. WHERE exactly? (refined box coordinates)
    """
    
    def __init__(self, in_channels=2048, num_classes=3):
        super(ROIHead, self).__init__()
        
        # ----------------------------------------------------------------
        # ROI Pooling: Extract fixed-size features from each proposal
        # WHY: Proposals are different sizes (some 50×50, some 200×100)
        #      Need same-size input for fully-connected layers
        # ----------------------------------------------------------------
        self.pool_size = 7  # Convert all proposals to 7×7 feature patches
        # WHY 7×7? Standard size - small enough to be fast, large enough to preserve detail
        
        roi_feature_size = 7 * 7 * in_channels  # 7 × 7 × 2048 = 100,352
        
        # ----------------------------------------------------------------
        # Fully Connected Layers: Process each proposal's features
        # WHY: Make final decision based on all information in the region
        # ----------------------------------------------------------------
        self.fc1 = nn.Linear(roi_feature_size, 1024)
        # WHY 1024? Compress 100K features to manageable size
        
        self.fc2 = nn.Linear(1024, 1024)
        # WHY second layer? Learn more complex patterns
        
        # ----------------------------------------------------------------
        # TWO FINAL OUTPUTS:
        # ----------------------------------------------------------------
        
        # Output 1: Classification - WHAT is this object?
        # WHY: Need to know if it's face/plate to decide blur strength
        self.classifier = nn.Linear(1024, num_classes)
        # Output: [background_prob, face_prob, plate_prob]
        
        # Output 2: Box Refinement - WHERE exactly is it?
        # WHY: RPN boxes are rough, need pixel-perfect accuracy for blurring
        self.box_refiner = nn.Linear(1024, num_classes * 4)
        # Output: 4 refined coordinates per class
        
    def forward(self, feature_maps, proposals):
        """
        Classify and refine each proposal
        
        Args:
            feature_maps: [1, 2048, 50, 38] from backbone (shared!)
            proposals: [1, 2000, 4] from RPN
            
        Returns:
            classes: [2000, 3] - probability for each class
            refined_boxes: [2000, 4] - precise coordinates
        """
        # ----------------------------------------------------------------
        # Extract features for each proposal from feature map
        # WHY: Instead of re-running CNN on each proposal (slow!),
        #      just extract relevant features from already-computed map
        # ----------------------------------------------------------------
        pooled_features = self._roi_pool(feature_maps, proposals)
        # For each of 2000 proposals, extract 7×7 feature patch
        # Output: [2000, 2048, 7, 7]
        
        # ----------------------------------------------------------------
        # Flatten for fully-connected layers
        # WHY: FC layers need 1D input, not 2D image-like features
        # ----------------------------------------------------------------
        pooled_features = pooled_features.flatten(start_dim=1)
        # Output: [2000, 100352]  (7×7×2048)
        
        # ----------------------------------------------------------------
        # Process through fully-connected layers
        # WHY: Learn to recognize patterns: "This 7×7 patch looks like a face"
        # ----------------------------------------------------------------
        x = torch.relu(self.fc1(pooled_features))  # [2000, 1024]
        x = torch.relu(self.fc2(x))                # [2000, 1024]
        
        # ----------------------------------------------------------------
        # OUTPUT 1: What is each proposal?
        # WHY: Need to know object type to decide: blur it or ignore it
        # ----------------------------------------------------------------
        class_logits = self.classifier(x)  # [2000, 3]
        class_probs = torch.softmax(class_logits, dim=1)
        # Example output for one proposal: [0.05, 0.92, 0.03]
        #                                   ^bg   ^face  ^plate
        # Interpretation: 92% confident this is a face!
        
        # ----------------------------------------------------------------
        # OUTPUT 2: Where exactly is each object?
        # WHY: RPN gave rough location, need pixel-perfect for clean blur
        # ----------------------------------------------------------------
        box_deltas = self.box_refiner(x)  # [2000, 12]  (3 classes × 4 coords)
        refined_boxes = self._apply_deltas(proposals, box_deltas)
        # Adjusts RPN boxes for tighter fit around actual object
        
        return class_probs, refined_boxes
    
    def _roi_pool(self, feature_maps, proposals):
        """
        Extract 7×7 features for each proposal box
        WHY: Proposals are different sizes, need uniform representation
        """
        # For each proposal [x1, y1, x2, y2]:
        # 1. Find corresponding region in feature_map
        # 2. Resize that region to 7×7
        # 3. Return pooled features
        
        # Simplified placeholder
        pooled = torch.rand(2000, 2048, 7, 7)
        return pooled
    
    def _apply_deltas(self, proposals, deltas):
        """
        Refine proposal boxes using predicted adjustments
        WHY: Convert learned adjustments to actual coordinates
        """
        # Apply deltas to proposals to get refined boxes
        refined = proposals  # Simplified
        return refined


# ============================================================================
# INFERENCE: Use trained model on Street View images
# ============================================================================

def detect_faces_and_plates(model, street_view_image):
    """
    Detect all faces and license plates in image
    
    WHY: Privacy compliance - must blur all identifiable information
    """
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():  # Don't compute gradients (faster)
        
        # Run through both stages
        class_probs, boxes = model(street_view_image)
        
        # Filter results
        detections = []
        for i in range(class_probs.size(0)):  # For each of 2000 proposals
            
            # Get predicted class
            class_id = torch.argmax(class_probs[i])
            confidence = class_probs[i, class_id]
            
            # Keep only confident detections (not background)
            # WHY 0.7 threshold? Trade-off:
            #   - Too low: blur random things (false positives OK)
            #   - Too high: miss some faces (false negatives BAD for privacy!)
            if class_id > 0 and confidence > 0.7:
                detections.append({
                    'class': 'face' if class_id == 1 else 'license_plate',
                    'box': boxes[i],
                    'confidence': confidence.item()
                })
        
        return detections


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize model
    print("Creating Faster R-CNN model...")
    model = FasterRCNN(num_classes=3)
    
    # Simulated Street View image
    street_view_image = torch.rand(1, 3, 800, 600)
    print(f"Input image shape: {street_view_image.shape}")
    
    # ----------------------------------------------------------------
    # Forward pass through entire network
    # ----------------------------------------------------------------
    print("\n--- STAGE 1: RPN ---")
    # Extract features
    feature_maps = model.backbone(street_view_image)
    print(f"Feature maps: {feature_maps.shape}")  # [1, 2048, 50, 38]
    
    # Get proposals
    proposals, scores = model.rpn(feature_maps)
    print(f"Proposals from RPN: {proposals.shape}")  # [1, 2000, 4]
    
    print("\n--- STAGE 2: ROI Head ---")
    # Classify proposals
    classes, boxes = model.roi_head(feature_maps, proposals)
    print(f"Class predictions: {classes.shape}")  # [2000, 3]
    print(f"Refined boxes: {boxes.shape}")        # [2000, 4]
    
    print("\n--- FINAL DETECTIONS ---")
    detections = detect_faces_and_plates(model, street_view_image)
    print(f"Found {len(detections)} faces/license plates to blur")
```

---

## 🎯 Key Takeaway
```
Raw Image (800×600)
    ↓ WHY: Need features, not pixels
BACKBONE (ResNet)
    ↓ Feature Maps (50×38×2048)
    ↓ WHY: Can't check all possible boxes
STAGE 1: RPN
    ↓ Top 2000 proposals
    ↓ WHY: Know WHAT they are
STAGE 2: ROI Head
    ↓ Final detections
BLUR THEM! ✅