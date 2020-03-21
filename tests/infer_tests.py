from glob import glob
import os
import pandas as pd
from pathlib import Path
import unittest
import shutil

from clouds.preprocess import Preprocessor
from clouds.inference import Inference
from clouds.experiments import GeneralInferExperiment
from clouds.experiments.utils import load_config

from .utils import load_paths_dict, download_weights


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
        download_weights()

    def tearDown(self):
        """Deleting the created files
        """
        shutil.rmtree(self.paths_dict["train_out"])
        shutil.rmtree(self.paths_dict["masks_out"])
        os.remove("fpn_resnet34_seg1_seed350_mvp_best.pth")

    def test_Inference_create_sub(self):
        """Testing that Inference.create_sub() runs smoothly.

        Not testing that it works properly.
        """
        # Seg only for now
        yml_path = "resources/configs/create_sub.yml"
        experiment_config = load_config(yml_path)
        exp = GeneralInferExperiment(experiment_config)

        infer = Inference(exp.model, exp.loaders["test"],
                          **experiment_config["infer_params"])
        out_df = infer.create_sub(sub=exp.sample_sub)
        print(out_df.head())
        self.assertTrue(isinstance(out_df, pd.DataFrame))


unittest.main(argv=[''], verbosity=2, exit=False)
