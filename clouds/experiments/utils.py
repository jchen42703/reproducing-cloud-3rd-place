import albumentations as albu
from clouds.io.custom_transforms import ToTensorV2
import os
import random
import numpy as np
import torch
import yaml


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


def load_weights(checkpoint_path, model):
    """Loads weights from a checkpoint.

    Args:
        checkpoint_path (str): path to a .pt or .pth checkpoint
        model (torch.nn.Module): <-

    Returns:
        Model with loaded weights and in train() mode

    """
    try:
        # catalyst weights
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state_dict["model_state_dict"]
    except:
        # anything else
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def load_config(yml_path):
    """Loads a .yml file.

    Args:
        yml_path (str): Path to a .yaml or .yml file.

    Returns:
        config (dict): parsed .yml config

    """
    with open(yml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config
