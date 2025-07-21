# DRIVE Semantic Segmentation Assignment Guide

This guide provides learning resources and implementation help for the DRIVE retinal vessel segmentation assignment.

## Overview

You will implement a semantic segmentation model to identify blood vessels in retinal fundus images using the DRIVE dataset. This involves creating dataloaders, designing a neural network architecture, training the model, and evaluating its performance.

---

## TODO 1: DataLoader Implementation

### Learning Resources
- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [PyTorch Dataset and DataLoader Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

### Implementation Tips
```python
# Basic structure for creating dataloaders
train_dataset = DRIVEDataset(data_dir, 'training')
test_dataset = DRIVEDataset(data_dir, 'testing')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

**Key Parameters:**
- `batch_size`: Start with 4-8 for this dataset
- `shuffle=True` for training, `shuffle=False` for testing
- `num_workers`: Use 2-4 for parallel data loading

---

## TODO 2: Model Architecture - U-Net

### What is U-Net?
U-Net is a convolutional neural network architecture specifically designed for biomedical image segmentation. It consists of:
- **Encoder (Contracting Path)**: Captures context through downsampling
- **Decoder (Expansive Path)**: Enables precise localization through upsampling
- **Skip Connections**: Combine low-level and high-level features

### Learning Resources
- [U-Net Original Paper](https://arxiv.org/abs/1505.04597) - Ronneberger et al., 2015
- [U-Net Architecture Explained](https://towardsdatascience.com/understanding-u-net-61276b10f360)
- [PyTorch U-Net Implementation Tutorial](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)
- [Interactive U-Net Visualization](https://poloclub.github.io/cnn-explainer/)

### Architecture Components
```python
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder blocks
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        # ... more encoder layers
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder blocks with skip connections
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)  # 1024 = 512 + 512 (skip)
        # ... more decoder layers
        
        # Final layer
        self.final = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

### Alternative Simpler Architectures
If U-Net seems too complex, consider:
- **Simple CNN Encoder-Decoder**: Basic convolutional layers with upsampling
- **FCN (Fully Convolutional Network)**: Replace fully connected layers with convolutional layers

---

## TODO 3: Training Loop Implementation

### Learning Resources
- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Loss Functions for Segmentation](https://pytorch.org/docs/stable/nn.html#loss-functions)

### Pseudocode
```
TRAINING_LOOP:
    Initialize model, optimizer, loss_function
    Initialize empty list for storing IoU scores
    
    FOR each epoch in num_epochs:
        Set model to training mode
        Initialize running_loss = 0
        Initialize running_iou = 0
        
        FOR each batch in train_loader:
            Load images, masks to device
            
            Zero gradients: optimizer.zero_grad()
            
            Forward pass: predictions = model(images)
            Calculate loss: loss = loss_function(predictions, masks)
            
            Backward pass: loss.backward()
            Update weights: optimizer.step()
            
            Update running_loss
            Calculate batch IoU and update running_iou
        
        Calculate epoch average loss and IoU
        Store IoU score for plotting
        Print progress
        
    RETURN list of IoU scores
```

### Implementation Tips
```python
def train_model(model, train_loader, num_epochs=20, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # or nn.BCEWithLogitsLoss() if no sigmoid in model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    iou_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_iou += calculate_iou(outputs, masks)
        
        epoch_iou = running_iou / len(train_loader)
        iou_scores.append(epoch_iou)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, IoU: {epoch_iou:.4f}')
    
    return iou_scores
```

---

## TODO 4: Intersection over Union (IoU) - Jaccard Index

### Mathematical Definition

The IoU measures the overlap between predicted and ground truth segmentation masks:

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

Where:
- $A$ = Predicted segmentation mask
- $B$ = Ground truth mask  
- $|A \cap B|$ = Intersection (pixels correctly predicted as vessel)
- $|A \cup B|$ = Union (all pixels predicted or actually vessel)

### IoU Range and Interpretation
- **Range**: [0, 1]
- **Perfect Match**: IoU = 1.0
- **No Overlap**: IoU = 0.0
- **Good Segmentation**: IoU > 0.7
- **Acceptable**: IoU > 0.5

### Learning Resources
- [IoU Explanation with Examples](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
- [Segmentation Metrics Overview](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)
- [Visual IoU Calculator](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

### Evaluation Pseudocode
```
EVALUATION:
    Set model to evaluation mode
    Initialize total_iou = 0
    Initialize num_samples = 0
    
    WITH torch.no_grad():
        FOR each batch in test_loader:
            Load images, masks to device
            Get predictions: outputs = model(images)
            
            FOR each sample in batch:
                Calculate IoU for this sample
                Add to total_iou
                Increment num_samples
    
    average_iou = total_iou / num_samples
    RETURN average_iou
```

---

## TODO 5: Visualization Implementation

### Training Curve Plotting
```python
def plot_training_curve(iou_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(iou_scores) + 1), iou_scores, 'b-', linewidth=2)
    plt.title('IoU Score During Training', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.show()
```

### Prediction Visualization Pseudocode
```
VISUALIZATION:
    Set model to evaluation mode
    Get one batch from test_loader
    
    WITH torch.no_grad():
        Get model predictions for the batch
    
    FOR sample_idx in range(num_samples):
        Extract: original_image, ground_truth, prediction
        Convert tensors to numpy for plotting
        
        Create subplot with 1 row, 3 columns:
            Column 1: Original image
            Column 2: Ground truth mask
            Column 3: Predicted mask
        
        Set appropriate titles and colormaps
        Remove axes for cleaner look
    
    Show or save the plot
```

### Implementation Example
```python
def visualize_predictions(model, test_loader, num_samples=1):
    model.eval()
    data_iter = iter(test_loader)
    images, masks = next(data_iter)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.to(device)
    
    with torch.no_grad():
        predictions = model(images)
    
    for i in range(min(num_samples, len(images))):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = images[i].cpu().permute(1, 2, 0)
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        gt = masks[i].cpu().squeeze()
        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        pred = predictions[i].cpu().squeeze()
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
```

---

## Additional Resources

### PyTorch Fundamentals
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch Book](https://pytorch.org/deep-learning-with-pytorch-book/)

### Computer Vision & Segmentation
- [CS231n Computer Vision Course](http://cs231n.stanford.edu/)
- [Medical Image Segmentation Tutorial](https://github.com/milesial/Pytorch-UNet)

### Loss Functions for Segmentation
- **Binary Cross-Entropy (BCE)**: Standard for binary segmentation
- **Dice Loss**: Good for imbalanced datasets
- **Focal Loss**: Handles hard examples better

### Tips for Success
1. **Start Simple**: Begin with a basic CNN before attempting U-Net
2. **Debug Step by Step**: Test each component individually
3. **Monitor Training**: Watch loss and IoU curves for signs of overfitting
4. **Experiment with Hyperparameters**: Learning rate, batch size, etc.
5. **Data Augmentation**: Consider rotations, flips for better generalization

### Common Issues and Solutions
- **Out of Memory**: Reduce batch size
- **Poor Performance**: Check data preprocessing and loss function
- **Overfitting**: Add dropout, reduce model complexity, or add regularization
- **Slow Training**: Use GPU acceleration, increase batch size if memory allows

Good luck with your implementation!