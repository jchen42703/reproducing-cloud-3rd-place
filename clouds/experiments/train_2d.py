import segmentation_models_pytorch as smp

from clouds.io import CloudDataset
from .train import TrainExperiment
from .utils import get_preprocessing, get_train_transforms, get_val_transforms


class TrainSegExperiment(TrainExperiment):
    """Stores the main parts of a segmentation experiment:

    - df split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    Note: There is no model_name for this experiment. There is `encoder` and
    `decoder` under `model_params`. You can also specify the attention_type
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_seg_yaml.py`
        """
        super().__init__(config=config)

    def get_datasets(self, train_ids, val_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        # encoder = self.model_params["encoder"]
        # preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder,
        #                                                      "imagenet")
        preprocessing_transform = get_preprocessing()
        train_aug = get_train_transforms(self.io_params["aug_key"])
        val_aug = get_val_transforms(self.io_params["aug_key"])
        dset_kwargs = {
            "data_folder": self.io_params["image_folder"],
            "masks_folder": self.io_params["masks_folder"],
            "preprocessing": preprocessing_transform
        }
        # creating the datasets
        train_dataset = CloudDataset(img_ids=train_ids, transforms=train_aug,
                                     **dset_kwargs)
        val_dataset = CloudDataset(img_ids=val_ids, transforms=val_aug,
                                   **dset_kwargs)
        return (train_dataset, val_dataset)

    def get_model(self):
        encoder = self.model_params["encoder"].lower()
        decoder = self.model_params["decoder"].lower()
        print(f"\nEncoder: {encoder}, Decoder: {decoder}")
        # setting up the seg model
        assert decoder in ["unet", "fpn"], \
            "`decoder` must be one of ['unet', 'fpn']"
        if decoder == "unet":
            model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                             classes=4, activation=None,
                             **self.model_params[decoder])
        elif decoder == "fpn":
            model = smp.FPN(encoder_name=encoder, encoder_weights="imagenet",
                            classes=4, activation=None,
                            **self.model_params[decoder])
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model
