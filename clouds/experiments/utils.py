import albumentations as albu
from clouds.io.custom_transforms import ToTensorV2
import os
import random
import numpy as np
import torch
from torch.jit import load
import yaml


class EnsembleModel(object):
    """
    Callable class for ensembled model inference
    """
    def __init__(self, models):
        self.models = models
        assert len(self.models) > 1

    def __call__(self, x):
        res = []
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)


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


def get_val_transforms(aug_key="mvp"):
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


def load_checkpoints(checkpoint_paths, models):
    """Loads checkpoints.

    Either loads a single checkpoint or loads an ensemble of checkpoints
    from `checkpoint_paths`

    Args:
        checkpoint_paths (list[str]): list of paths to checkpoints
        models (list[torch.nn.Module]): models corresponding to the
            checkpoint_paths. 
            If it is left as [], the function assumes that the weights
            are traced (so no models are needed).

    Returns:
        model (torch.nn.Module): The single/ensembled model

    """
    assert isinstance(checkpoint_paths, list), \
        "Make sure checkpoint_paths is specified in list format in the\
        yaml file."
    assert isinstance(models, list), \
        "Make sure model_names is specified in list format in the\
        yaml file."
    if len(models) != 0:
        assert len(checkpoint_paths) == len(models), \
            "The number of checkpoints and models should be the same."

    # single model instances
    if len(checkpoint_paths) == 1:
        try:
            # loading traceable
            model = load(checkpoint_paths[0]).cuda().eval()
            print(f"Traced model from {checkpoint_paths}")
        except:
            model = load_weights(checkpoint_paths[0],
                                 models[0]).cuda().eval()
            print(f"Loaded model from {checkpoint_paths}")
    # ensembled models
    elif len(checkpoint_paths) > 1:
        try:
            model = EnsembleModel([load(path).cuda().eval()
                                   for path in checkpoint_paths])
        except:
            model = EnsembleModel([load_weights(path, model).cuda().eval()
                                   for (path, model) in
                                   zip(checkpoint_paths, models)])
        print(f"Loaded an ensemble from {checkpoint_paths}")
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
