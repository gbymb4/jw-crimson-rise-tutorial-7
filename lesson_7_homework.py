import os
import requests
import py7zr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class DRIVEDataset(Dataset):
    """
    Custom Dataset class for DRIVE retinal fundus imaging dataset.
    Handles loading, preprocessing, and serving image-mask pairs.
    """
    
    def __init__(self, data_dir: str, split: str = 'training', transform=None):
        """
        Initialize the DRIVE dataset.
        
        Args:
            data_dir (str): Path to the data directory containing DRIVE folder
            split (str): Either 'training' or 'testing'
            transform: Optional transforms to apply to images
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Set up paths - handle both "testing" input and "test" folder name
        drive_base = os.path.join(data_dir, 'DRIVE')
        
        # The dataset folder is actually named "test", not "testing"
        actual_split = 'test' if split == 'testing' else split
        self.images_dir = os.path.join(drive_base, actual_split, 'images')
        
        # Try different possible mask directory structures
        # Based on the user's folder structure, prioritize the correct paths
        if split == 'training':
            possible_mask_dirs = [
                os.path.join(drive_base, split, '1st_manual'),
                os.path.join(drive_base, split, 'mask'),
                os.path.join(drive_base, split, 'masks'),
                os.path.join(drive_base, split, 'manual'),
                os.path.join(drive_base, split)
            ]
        else:  # test or testing
            # Handle both "test" and "testing" folder names
            test_split = 'test' if split == 'testing' else split
            possible_mask_dirs = [
                os.path.join(drive_base, test_split, 'mask'),
                os.path.join(drive_base, test_split, '1st_manual'),
                os.path.join(drive_base, test_split, 'masks'),
                os.path.join(drive_base, test_split, 'manual'),
                os.path.join(drive_base, test_split)
            ]
        
        self.masks_dir = None
        for mask_dir in possible_mask_dirs:
            if os.path.exists(mask_dir):
                # Check if there are any mask files in this directory
                if any(f.endswith(('.gif', '.png', '.tif', '.jpg', '.jpeg')) for f in os.listdir(mask_dir) 
                       if 'manual' in f.lower() or 'mask' in f.lower() or 'gt' in f.lower()):
                    self.masks_dir = mask_dir
                    print(f"Found mask directory: {self.masks_dir}")
                    break
        
        if self.masks_dir is None:
            print(f"Warning: No mask directory found. Searched in: {possible_mask_dirs}")
            print(f"Available directories in {os.path.join(drive_base, split)}:")
            if os.path.exists(os.path.join(drive_base, split)):
                for item in os.listdir(os.path.join(drive_base, split)):
                    print(f"  - {item}")
        
        # Get all image files and their corresponding masks
        self.image_files = []
        self.mask_files = []
        
        if os.path.exists(self.images_dir):
            print(f"Looking for images in: {self.images_dir}")
            for file in sorted(os.listdir(self.images_dir)):
                if file.endswith(('.tif', '.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(self.images_dir, file)
                    self.image_files.append(image_path)
                    
                    # Find corresponding mask file
                    mask_path = self._find_mask_file(file)
                    if mask_path:
                        self.mask_files.append(mask_path)
                        print(f"Matched: {file} -> {os.path.basename(mask_path)}")
                    else:
                        print(f"Warning: No mask found for image {file}")
        
        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks in {split} set")
        
        # Only keep pairs where both image and mask exist
        if len(self.image_files) != len(self.mask_files):
            min_len = min(len(self.image_files), len(self.mask_files))
            self.image_files = self.image_files[:min_len]
            self.mask_files = self.mask_files[:min_len]
            print(f"Adjusted to {len(self.image_files)} matched image-mask pairs")
    
    def _find_mask_file(self, image_filename: str) -> str:
        """
        Find the corresponding mask file for a given image filename.
        
        Args:
            image_filename: Name of the image file
            
        Returns:
            Path to the corresponding mask file, or None if not found
        """
        if self.masks_dir is None:
            return None
            
        # Extract base name (without extension) and extract the number
        base_name = os.path.splitext(image_filename)[0]
        
        # Extract the number from the filename
        # For training: "21_training.tif" -> "21"
        # For testing: "01_test.tif" -> "01"
        if '_' in base_name:
            number_part = base_name.split('_')[0]
        else:
            number_part = base_name
        
        # Possible mask naming patterns based on the dataset structure
        if self.split == 'training':
            # Training masks are in format: XX_manual1.gif
            possible_patterns = [
                f"{number_part}_manual1.gif",
                f"{number_part}_manual.gif",
                f"{base_name}_manual1.gif",
                f"{base_name}_manual.gif",
                f"{number_part}.gif",
                f"{base_name}.gif"
            ]
        else:
            # Test masks are in format: XX_test_mask.gif
            possible_patterns = [
                f"{number_part}_test_mask.gif",
                f"{base_name}_mask.gif",
                f"{number_part}_mask.gif",
                f"{base_name}.gif",
                f"{number_part}.gif"
            ]
        
        # Try each pattern
        for pattern in possible_patterns:
            mask_path = os.path.join(self.masks_dir, pattern)
            if os.path.exists(mask_path):
                return mask_path
        
        # If no exact match, search for files containing the number part
        if os.path.exists(self.masks_dir):
            for mask_file in os.listdir(self.masks_dir):
                if (number_part in mask_file and 
                    mask_file.endswith(('.gif', '.png', '.tif', '.jpg', '.jpeg')) and
                    ('manual' in mask_file.lower() or 'mask' in mask_file.lower() or 'gt' in mask_file.lower())):
                    return os.path.join(self.masks_dir, mask_file)
        
        return None
    
    def __len__(self):
        return min(len(self.image_files), len(self.mask_files))
    
    def __getitem__(self, idx):
        """
        Get a single image-mask pair.
        
        Returns:
            tuple: (image_tensor, mask_tensor) both resized to 256x256
        """
        if idx >= len(self.mask_files):
            raise IndexError(f"Index {idx} out of range. Only {len(self.mask_files)} image-mask pairs available.")
        
        # Load image and mask
        image = Image.open(self.image_files[idx]).convert('RGB')
        mask = Image.open(self.mask_files[idx]).convert('L')  # Grayscale
        
        # Resize with padding to maintain aspect ratio
        image_tensor = self._resize_with_padding(image, (256, 256))
        mask_tensor = self._resize_with_padding(mask, (256, 256), is_mask=True)
        
        # Convert to tensors
        image_tensor = transforms.ToTensor()(image_tensor)
        mask_tensor = transforms.ToTensor()(mask_tensor)
        
        # Normalize image to [0, 1]
        image_tensor = image_tensor.float()
        
        # Convert mask to binary (0 or 1)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor
    
    def _resize_with_padding(self, image, target_size: Tuple[int, int], is_mask: bool = False):
        """
        Resize image with padding to maintain aspect ratio.
        
        Args:
            image: PIL Image
            target_size: Target (width, height)
            is_mask: Whether this is a mask (affects interpolation method)
        """
        original_size = image.size
        target_w, target_h = target_size
        
        # Calculate scaling factor to fit within target size
        scale = min(target_w / original_size[0], target_h / original_size[1])
        
        # Calculate new size
        new_w = int(original_size[0] * scale)
        new_h = int(original_size[1] * scale)
        
        # Resize image
        if is_mask:
            resized_image = image.resize((new_w, new_h), Image.NEAREST)
        else:
            resized_image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Create new image with target size and paste resized image
        if is_mask:
            new_image = Image.new('L', target_size, 0)  # Black background for mask
        else:
            new_image = Image.new('RGB', target_size, (0, 0, 0))  # Black background
        
        # Calculate padding
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        new_image.paste(resized_image, (paste_x, paste_y))
        
        return new_image
    

def download_and_extract_data(url: str, data_dir: str = 'data'):
    """
    Download and extract the DRIVE dataset from the provided URL.
    
    Args:
        url (str): URL to download the .7z file
        data_dir (str): Directory to extract data to
    """
    os.makedirs(data_dir, exist_ok=True)
    
    archive_path = os.path.join(data_dir, 'drive_dataset.7z')
    
    # Download if not already present
    if not os.path.exists(archive_path):
        print("Downloading DRIVE dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(archive_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed!")
    
    # Extract if DRIVE folder doesn't exist
    drive_path = os.path.join(data_dir, 'DRIVE')
    if not os.path.exists(drive_path):
        print("Extracting dataset...")
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extractall(path=data_dir)
        print("Extraction completed!")
    
    return data_dir


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) / Jaccard Index.
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0  # Perfect match when both are empty
    
    return (intersection / union).item()


# =============================================================================
# TODO SECTIONS FOR STUDENTS
# =============================================================================

# TODO 1: Implement DataLoader
def create_dataloaders(data_dir: str, batch_size: int = 4, num_workers: int = 2):
    """
    TODO: Create training and testing dataloaders.
    
    Instructions:
    1. Create training and testing datasets using DRIVEDataset class
    2. Wrap them in DataLoader with appropriate parameters
    3. Consider shuffle=True for training, shuffle=False for testing
    4. Return both dataloaders
    
    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Your code here
    raise NotImplementedError("Student needs to implement dataloader creation")


# TODO 2: Implement Model Architecture
class SegmentationModel(nn.Module):
    """
    TODO: Implement a neural network for semantic segmentation.
    
    Instructions:
    1. Design a model that takes 3-channel RGB images (3, 256, 256)
    2. Output should be single-channel segmentation masks (1, 256, 256)
    3. Consider using:
       - Encoder-decoder architecture (U-Net style)
       - Skip connections
       - Appropriate activation functions
    4. Use nn.Sigmoid() as final activation for binary segmentation
    
    Suggested architectures:
    - Simple CNN with upsampling
    - U-Net
    - DeepLabV3 (if you want a challenge)
    """
    
    def __init__(self):
        super(SegmentationModel, self).__init__()
        # Your model definition here
        raise NotImplementedError("Student needs to implement model architecture")
    
    def forward(self, x):
        # Your forward pass here
        raise NotImplementedError("Student needs to implement forward pass")


# TODO 3: Implement Training Loop
def train_model(model, train_loader, num_epochs: int = 20, learning_rate: float = 0.001):
    """
    TODO: Implement the training loop.
    
    Instructions:
    1. Use appropriate loss function (BCELoss or BCEWithLogitsLoss)
    2. Use appropriate optimizer (Adam recommended)
    3. Track IoU scores during training for plotting
    4. Print progress every few epochs
    5. Return list of IoU scores for plotting
    
    Args:
        model: The segmentation model
        train_loader: Training dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        list: IoU scores for each epoch
    """
    # Your training code here
    raise NotImplementedError("Student needs to implement training loop")


# TODO 4: Implement Testing/Evaluation
def evaluate_model(model, test_loader):
    """
    TODO: Implement model evaluation.
    
    Instructions:
    1. Set model to evaluation mode
    2. Disable gradients during evaluation
    3. Calculate average IoU over the test set
    4. Return average IoU score
    
    Args:
        model: Trained segmentation model
        test_loader: Test dataloader
    
    Returns:
        float: Average IoU score on test set
    """
    # Your evaluation code here
    raise NotImplementedError("Student needs to implement evaluation")


# TODO 5: Implement Visualization Functions
def plot_training_curve(iou_scores: List[float]):
    """
    TODO: Plot the IoU scores during training.
    
    Instructions:
    1. Create a line plot of IoU scores vs epochs
    2. Add appropriate labels and title
    3. Save or display the plot
    
    Args:
        iou_scores: List of IoU scores from training
    """
    # Your plotting code here
    raise NotImplementedError("Student needs to implement training curve plotting")


def visualize_predictions(model, test_loader, num_samples: int = 1):
    """
    TODO: Visualize model predictions.
    
    Instructions:
    1. Get a sample from the test loader
    2. Make prediction using the model
    3. Create side-by-side plot showing:
       - Original image
       - Ground truth mask
       - Predicted mask
    4. Use appropriate colormaps and titles
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        num_samples: Number of samples to visualize
    """
    # Your visualization code here
    raise NotImplementedError("Student needs to implement prediction visualization")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def explore_dataset_structure(data_dir: str):
    """
    Helper function to explore and understand the dataset structure.
    Useful for debugging when masks are not found.
    """
    drive_path = os.path.join(data_dir, 'DRIVE')
    if not os.path.exists(drive_path):
        print(f"DRIVE folder not found in {data_dir}")
        return
    
    print("DRIVE Dataset Structure:")
    print("=" * 40)
    
    for root, dirs, files in os.walk(drive_path):
        level = root.replace(drive_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Show first few files in each directory
        subindent = ' ' * 2 * (level + 1)
        for i, file in enumerate(files[:5]):  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
        print()


def main():
    """
    Main execution function. This part is completed for you.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download and extract data
    data_url = "https://data.mendeley.com/public-files/datasets/frv89hjgrr/files/f61c5f08-f18d-4206-8416-a4c8a69b3fce/file_downloaded"
    data_dir = download_and_extract_data(data_url)
    
    # Explore dataset structure for debugging
    print("\nExploring dataset structure...")
    explore_dataset_structure(data_dir)
    
    # Test dataset loading (this part works)
    print("\nTesting dataset loading...")
    train_dataset = DRIVEDataset(data_dir, 'training')
    test_dataset = DRIVEDataset(data_dir, 'testing')
    
    # Sample one item to verify (only if dataset has items)
    if len(train_dataset) > 0:
        sample_image, sample_mask = train_dataset[0]
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample mask shape: {sample_mask.shape}")
        print(f"Image value range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
        print(f"Mask value range: [{sample_mask.min():.3f}, {sample_mask.max():.3f}]")
        
        print("\n" + "="*50)
        print("DATASET LOADING COMPLETED SUCCESSFULLY!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("DATASET LOADING FAILED!")
        print("Please check the dataset structure above and verify mask files exist.")
        print("="*50)
        return
    
    print("\nNow complete the TODO sections:")
    print("1. Create dataloaders")
    print("2. Implement the segmentation model")
    print("3. Implement the training loop")
    print("4. Implement evaluation")
    print("5. Create visualization functions")
    print("\nThen uncomment the code below to run training:")
    
    # Uncomment these lines after implementing the TODO sections:
    '''
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(data_dir)
    
    # Create and train model
    model = SegmentationModel().to(device)
    iou_scores = train_model(model, train_loader)
    
    # Evaluate model
    test_iou = evaluate_model(model, test_loader)
    print(f"Final test IoU: {test_iou:.4f}")
    
    # Plot results
    plot_training_curve(iou_scores)
    visualize_predictions(model, test_loader)
    '''


if __name__ == "__main__":
    main()
    
    