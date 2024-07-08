import os
from pathlib import Path
import re
import random
from sklearn.model_selection import KFold
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def load_image_paths(directory):
    """Loads all image paths from the specified directory."""
    path = Path(directory)
    image_paths = list(path.glob('*.jpg'))
    return [str(img) for img in image_paths]

def group_images_by_prefix(image_paths):
    """Groups images by their prefix NUMSUBJECT_NUMACTIVITY_NUM_ROUTINE."""
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)_\d+_\d+.jpg')
    grouped = {}
    for img_path in image_paths:
        match = pattern.search(os.path.basename(img_path))
        if match:
            prefix = f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
            if prefix not in grouped:
                grouped[prefix] = []
            grouped[prefix].append(img_path)
    return list(grouped.values())

def create_dataloaders(image_groups, n_splits=5, batch_size=32, transform=None):
    """Creates DataLoaders for cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    dataloaders = []
    
    for train_index, val_index in kf.split(image_groups):
        train_images = [img for i in train_index for img in image_groups[i]]
        val_images = [img for i in val_index for img in image_groups[i]]
        
        train_dataset = CustomImageDataset(train_images, transform)
        val_dataset = CustomImageDataset(val_images, transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        dataloaders.append((train_loader, val_loader))
    
    return dataloaders

def main():
    image_directory = "C:/Users/iagor/Documents/git/data-definer/out/"
    batch_size = 32
    n_splits = 5
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image_paths = load_image_paths(image_directory)
    image_groups = group_images_by_prefix(image_paths)
    dataloaders = create_dataloaders(image_groups, n_splits, batch_size, transform)
    
    for fold, (train_loader, val_loader) in enumerate(dataloaders):
        print(f"Fold {fold+1}")
        for images in train_loader:
            print(f"Train batch: {images.size()}")
        for images in val_loader:
            print(f"Validation batch: {images.size()}")

if __name__ == "__main__":
    main()
