from albumentations.core.transforms_interface import BasicTransform
import torch


class ToTensorV2(BasicTransform):
    """Converts both the image and mask to Tensors.

    Convert image and mask to `torch.Tensor`. This is different from the
    albumentations version in that it also transposes the mask instead of
    just the image.

    """

    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask.transpose(2, 0, 1))

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}
