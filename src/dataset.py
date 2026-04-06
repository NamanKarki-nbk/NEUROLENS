import os 
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


train_transform = A.Compose([
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


class BrainTumorDataset(Dataset):
    
    def __init__(self, image_dir, transforms=None):
        self.image_paths = []
        self.labels = []
        self.transforms = transforms
        
        classes = sorted(os.listdir(image_dir))
        self.classes = classes
        
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(image_dir, cls)
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(label)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Corrupted image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
        
        label = self.labels[index]
        return img, label
    
def get_dataloader(image_dir, batch_size=32, num_workers=4, transforms=None):
    dataset = BrainTumorDataset(image_dir, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader