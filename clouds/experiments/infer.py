from abc import abstractmethod
from pathlib import Path
import pandas as pd

import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from clouds.io import TestCloudDataset
from .utils import get_val_transforms, get_preprocessing


class InferExperiment(object):
    def __init__(self, config: dict):
        """
        Args:
            config (dict):
        Attributes:
            config-related:
                config (dict):
                io_params (dict):
                    in_dir (key: str): path to the data folder
                    test_size (key: float): split size for test
                    split_seed (key: int): seed
                    batch_size (key: int): <-
                    num_workers (key: int): # of workers for data loaders
            split_dict (dict): test_ids
            test_dset (torch.data.Dataset): <-
            loaders (dict): train/validation loaders
            model (torch.nn.Module): <-
        """
        # for reuse
        self.config = config
        self.io_params = config["io_params"]
        self.model_params = config["model_params"]
        # initializing the experiment components
        self.sample_sub = self.setup_df()
        self.test_ids = self.get_test_ids()
        self.test_dset = self.get_datasets(self.test_ids)
        self.loaders = self.get_loaders()
        self.models = self.get_models()

    @abstractmethod
    def get_datasets(self, test_ids):
        """
        Initializes the data augmentation and preprocessing transforms. Creates
        and returns the train and validation datasets.
        """
        return

    @abstractmethod
    def get_models(self):
        """
        Creates and returns the models to infer (and ensemble). Note that
        this differs from TrainExperiment variants because they only fetch
        one model.
        """
        return

    def setup_df(self):
        """
        Setting up the dataframe to have the `im_id` & `label` columns;
            im_id: the base img name
            label: the label name
        """
        sample_sub_csv_path = self.config["sample_sub_csv_path"]
        sub = pd.read_csv(sample_sub_csv_path)
        sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
        sub["img_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])
        return sub

    def get_test_ids(self):
        """
        Returns the test image ids.
        """
        image_labels = self.sample_sub["Image_Label"]
        test_ids = image_labels.apply(lambda x:
                                      x.split("_")[0]).drop_duplicates().values
        print(f"Number of test ids: {len(test_ids)}")
        test_ids = [f"{Path(fname).stem}.npy" for fname in test_ids]
        return test_ids

    def get_loaders(self):
        """
        Creates train/val loaders from datasets created in self.get_datasets.
        Returns the loaders.
        """
        # setting up the loaders
        batch_size = self.io_params["batch_size"]
        num_workers = self.io_params["num_workers"]
        test_loader = DataLoader(self.test_dset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
        return {"test": test_loader}


class GeneralInferExperiment(InferExperiment):
    def __init__(self, config: dict):
        """
        Args:
            config (dict):
        Attributes:
            config-related:
                config (dict):
                io_params (dict):
                    in_dir (key: str): path to the data folder
                    test_size (key: float): split size for test
                    split_seed (key: int): seed
                    batch_size (key: int): <-
                    num_workers (key: int): # of workers for data loaders
            split_dict (dict): test_ids
            test_dset (torch.data.Dataset): <-
            loaders (dict): train/validation loaders
            model (torch.nn.Module): <-
        """
        self.encoders = config["model_params"].get("encoders")
        self.decoders = config["model_params"].get("decoders")
        super().__init__(config=config)

    def get_datasets(self, test_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        # encoder = self.model_params["encoder"]
        # preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder,
        #                                                      "imagenet")
        preprocessing_transform = get_preprocessing()
        val_aug = get_val_transforms(self.io_params["aug_key"])
        # Using kwargs to meet line length (flake8)
        dset_kwargs = {
            "data_folder": self.io_params["image_folder"],
            "transforms": val_aug,
            "preprocessing": preprocessing_transform
        }
        test_dset = TestCloudDataset(img_ids=test_ids, **dset_kwargs)
        return test_dset

    def get_models(self):
        """
        Fetches multiple models as a list. If it's a single model, `models`
        will be a length one list.
        """
        pairs = list(zip(self.encoders, self.decoders))
        print(f"Models: {pairs}")
        # setting up the seg model
        models = [smp.__dict__[decoder](encoder_name=encoder,
                                        encoder_weights=None,
                                        classes=4, activation=None,
                                        **self.model_params[decoder])
                  for encoder, decoder in pairs]
        return models
