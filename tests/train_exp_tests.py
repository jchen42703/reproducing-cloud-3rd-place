from glob import glob
import os
import pandas as pd
from pathlib import Path
import unittest
import shutil
import torch

from clouds.preprocess import Preprocessor
from clouds.experiments import TrainSegExperiment
from clouds.experiments.utils import load_config


class TrainExperimentsTests(unittest.TestCase):
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

        preprocess_yml_path = "resources/configs/create_dset.yml"
        preprocess_config = load_config(preprocess_yml_path)
        self.paths_dict = load_paths_dict(preprocess_config)

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

    def tearDown(self):
        """Deleting the created files
        """
        shutil.rmtree(self.paths_dict["train_out"])
        shutil.rmtree(self.paths_dict["masks_out"])

    def test_TrainSegExperiment(self):
        """Testing that TrainSegExperiment contains the correct components.
        """
        # Seg only for now
        yml_path = "resources/configs/train_seg.yml"
        experiment_config = load_config(yml_path)
        exp = TrainSegExperiment(experiment_config)

        # type checking
        self.assertTrue(isinstance(exp.opt, torch.optim.Optimizer))
        self.assertTrue(isinstance(exp.lr_scheduler,
                                   torch.optim.lr_scheduler._LRScheduler))
        self.assertTrue(isinstance(exp.train_dset, torch.utils.data.Dataset))
        self.assertTrue(isinstance(exp.val_dset, torch.utils.data.Dataset))
        self.assertTrue(isinstance(exp.model, torch.nn.Module))

    def test_TrainSegExperiment_Stage1(self):
        """Testing that TrainSegExperiment for Stage 1
        """
        pass

    @unittest.skip("Will take time to download weights.")
    def test_TrainSegExperiment_from_checkpoint(self):
        """Testing that TrainSegExperiment loads weights properly
        """
        pass


def load_paths_dict(preprocess_config):
    """Creates a dictionary of paths without the path to the df.

    This is so that the attributes for Preprocessor can be set recursively.

    Args:
        preprocess_config (dict): From loading 'create_dset.yml'
    Returns:
        paths_dict (dict): same as config['paths_params'] but without the
            'train_csv_path'

    """
    paths_params = preprocess_config["paths_params"]
    paths_dict = {
        "train_dir": paths_params["train_dir"],
        "test_dir": paths_params["test_dir"],
        "train_out": paths_params["train_out"],
        "test_out": paths_params["test_out"],
        "masks_out": paths_params["masks_out"],
    }
    return paths_dict


unittest.main(argv=[''], verbosity=2, exit=False)
