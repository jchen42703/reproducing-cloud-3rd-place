from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import yaml
import unittest
import shutil
import torch

from clouds.preprocess import Preprocessor
from clouds.io import CloudDataset
from clouds.experiments import get_train_transforms, get_preprocessing


class DatasetTests(unittest.TestCase):
    """Testing preprocessing procedures
    """
    def setUp(self):
        """Initializing the parameters:
        """
        try:
            self.img_names = [Path(fpath).name
                              for fpath in glob("resources/*.jpg")]
            self.rle_df = pd.read_csv("resources/train_sample.csv")
        except FileNotFoundError:
            raise Exception("Please make sure to run tests within the",
                            "test directory.")
        # loading the config
        yml_path = "resources/configs/create_dset.yml"
        with open(yml_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        paths_params = config["paths_params"]
        self.paths_dict = {
            "train_dir": paths_params["train_dir"],
            "test_dir": paths_params["test_dir"],
            "train_out": paths_params["train_out"],
            "test_out": paths_params["test_out"],
            "masks_out": paths_params["masks_out"],
        }

        # Creates the directory if it does not already exist
        for dir_ in ["train_out", "test_out", "masks_out"]:
            # so it doesn't create the test dir (None)
            dir_of_interest = self.paths_dict[dir_]
            if dir_of_interest is not None:
                if not os.path.isdir(dir_of_interest):
                    os.mkdir(dir_of_interest)

        # Sets up the preprocessed sample to load from
        self.out_shape_cv2 = (576, 384)
        self.preprocessor = Preprocessor(self.rle_df, self.paths_dict,
                                         self.out_shape_cv2)
        self.preprocessor.execute_all()

        arr_search_path = os.path.join(self.paths_dict["train_out"], "*.npy")
        self.arr_names = [Path(fpath).name for fpath in glob(arr_search_path)]

    def tearDown(self):
        """Deleting the created files
        """
        shutil.rmtree(self.paths_dict["train_out"])
        shutil.rmtree(self.paths_dict["masks_out"])

    def test_CloudDataset_no_transforms(self):
        """Testing CloudDataset with no transforms
        """
        dset = CloudDataset(self.arr_names, self.paths_dict["train_out"],
                            self.paths_dict["masks_out"])

        for idx in range(len(dset)):
            x, y = dset[idx]
            # shape checks
            self.assertEqual(x.shape, (384, 576, 3))
            self.assertEqual(y.shape, (384, 576, 4))
            # type checks
            self.assertEqual(x.dtype, np.uint8)
            # Note: CloudDataset converts masks to np.float32
            self.assertEqual(y.dtype, np.float32)

    def test_CloudDataset_with_transforms(self):
        """Testing CloudDataset with transforms.

        Take transforms from `experiments/utils.py`
        """
        train_t = get_train_transforms(aug_key="mvp")
        preprocess_t = get_preprocessing()
        dset = CloudDataset(self.arr_names, self.paths_dict["train_out"],
                            self.paths_dict["masks_out"], transforms=train_t,
                            preprocessing=preprocess_t)

        for idx in range(len(dset)):
            x, y = dset[idx]
            # shape checks
            self.assertEqual(x.shape, (3, 384, 576))
            self.assertEqual(y.shape, (4, 384, 576))
            # type checks
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.float32)


unittest.main(argv=[''], verbosity=2, exit=False)
