import albumentations as A
from albumentations.pytorch import ToTensorV2

# For training
train_transform = A.Compose([
    A.Resize(320, 320),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# For testing/validation
val_transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(),
    ToTensorV2()
])
