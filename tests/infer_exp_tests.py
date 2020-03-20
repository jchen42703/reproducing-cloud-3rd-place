from glob import glob
import os
import pandas as pd
from pathlib import Path
import unittest
import shutil
import torch

import segmentation_models_pytorch as smp

from clouds.preprocess import Preprocessor
from clouds.experiments import GeneralInferExperiment
from clouds.experiments.utils import load_config


class InferExperimentsTests(unittest.TestCase):
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

        # Inference specific prep
        download_weights()

    def tearDown(self):
        """Deleting the created files
        """
        shutil.rmtree(self.paths_dict["train_out"])
        shutil.rmtree(self.paths_dict["masks_out"])
        os.remove("fpn_resnet34_seg1_seed350_mvp_best.pth")

    def test_GeneralInferExperiment(self):
        """Testing that GeneralInferExperiment contains the correct components.
        """
        # Seg only for now
        yml_path = "resources/configs/create_sub.yml"
        experiment_config = load_config(yml_path)
        exp = GeneralInferExperiment(experiment_config)

        # type checking
        self.assertTrue(isinstance(exp.test_ids, list))
        # Test ids is all test images
        self.assertEqual(len(exp.test_ids), len(exp.sample_sub)/4)
        self.assertTrue(isinstance(exp.test_dset, torch.utils.data.Dataset))
        self.assertTrue(isinstance(exp.loaders["test"],
                                   torch.utils.data.DataLoader))
        self.assertTrue(isinstance(exp.model, torch.nn.Module))

        # value checking
        # Models (only ResNet34 + FPN in this case)
        # Note: GeneralInferExperiment doesn't actually load the weights
        # so this should just be the regular weights
        # This actually might fail because random weight initialization
        # and I didn't seed...
        # I should just move the weight loading to GeneralInferExperiment lol
        model1 = exp.models[0]
        model2 = smp.FPN(encoder_name="resnet34",
                         encoder_weights=None,
                         classes=4, activation=None,
                         decoder_dropout=0.2)
        checkpoint_path = "fpn_resnet34_seg1_seed350_mvp_best.pth"
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state_dict["model_state_dict"]
        model2.load_state_dict(state_dict, strict=True)
        model2 = model2.cuda().eval()

        self.assertTrue(test_model_equal(model1, model2))

    # def test_GeneralInferExperiment_Stage1(self):
    #     """Testing that GeneralInferExperiment for Stage 1
    #     """
    #     pass

    # @unittest.skip("Will take time to download weights.")
    # def test_GeneralInferExperiment_from_checkpoint(self):
    #     """Testing that GeneralInferExperiment loads weights properly
    #     """
    #     pass


def test_model_equal(model_1, model_2):
    """Tests if two models are equal or not.
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(),
                                      model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        return True
    else:
        return False


def download_sample_sub():
    """Downloads sample_submission.csv for tests.
    """
    from google_drive_downloader import GoogleDriveDownloader as gdd
    weights_id = "1-2ON7yRtpYX62yvnZ7tCb-WJs0_mOUqL"
    save_path = "./sample_submission.csv"
    gdd.download_file_from_google_drive(file_id=weights_id,
                                        dest_path=save_path,
                                        unzip=False)


def download_weights():
    """Downloads example weights for tests.
    """
    from google_drive_downloader import GoogleDriveDownloader as gdd
    weights_id = "1YMewsRkJoybsy4Qs05UJG_0ZvgL8pZg5"
    save_path = "./fpn_resnet34_seg1_seed350_mvp_best.pth"
    gdd.download_file_from_google_drive(file_id=weights_id,
                                        dest_path=save_path,
                                        unzip=False)


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
