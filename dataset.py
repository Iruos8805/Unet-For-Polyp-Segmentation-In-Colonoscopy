import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from transform import *


# location
X_folder = './kvasir-seg/Kvasir-SEG/images'
Y_folder = './kvasir-seg/Kvasir-SEG/masks'


#creating a list
all_in = [f for f in os.listdir(X_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
all_out = [f for f in os.listdir(Y_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Getting full paths 
image_paths = [os.path.join(X_folder, f) for f in all_in]
mask_paths = [os.path.join(Y_folder, f) for f in all_out]

#rrain test split
X_train, X_test, y_train, y_test = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)


#dataset defined
class CustomDataset(Dataset):
    def __init__(self,image_path,mask_path, transform = None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self,idx):
        image = np.array(Image.open(self.image_path[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_path[idx]).convert('L'))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()/255.0
        
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)/255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)/255.0

        return image,mask
