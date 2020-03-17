import albumentations as albu
from clouds.io.custom_transforms import ToTensorV2
import os
import random
import numpy as np
import torch


def get_train_transforms(aug_key="mvp"):
    """Training transforms
    """
    transform_dict = {
        "mvp": [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30,
                                  shift_limit=0, p=0.5, border_mode=0),
            albu.GridDistortion(p=0.5),
        ],
    }
    train_transform = transform_dict[aug_key]
    return albu.Compose(train_transform)


def get_valid_transforms(aug_key="mvp"):
    """Validation transforms
    """
    transform_dict = {
        "mvp": [],
                     }
    test_transform = transform_dict[aug_key]
    return albu.Compose(test_transform)


def get_preprocessing():
    """Construct preprocessing transform

    Normalizes using the torchvision stats.
    https://pytorch.org/docs/stable/torchvision/models.html
    Also, converts to Tensor.

    Args:

    Return:
        transform: albumentations.Compose

    """
    transform_list = [
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0, p=1),
        ToTensorV2(),
    ]
    return albu.Compose(transform_list)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # uses the inbuilt cudnn auto-tuner to find the fastest convolution
    # algorithms. -
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
