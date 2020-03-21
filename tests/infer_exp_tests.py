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

from .utils import load_paths_dict, download_weights, test_model_equal


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


unittest.main(argv=[''], verbosity=2, exit=False)
