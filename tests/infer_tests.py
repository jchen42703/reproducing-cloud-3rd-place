from glob import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path
import unittest
import shutil
import matplotlib.pyplot as plt

from clouds.preprocess import Preprocessor
from clouds.inference import Inference
from clouds.experiments import GeneralInferExperiment
from clouds.experiments.utils import load_config
from clouds.preprocess import make_mask

from utils import load_paths_dict, download_weights


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
        os.remove("submission.csv")

    def test_Inference_create_sub(self):
        """Testing that Inference.create_sub() runs smoothly.
        """
        # Seg only for now
        yml_path = "resources/configs/create_sub.yml"
        experiment_config = load_config(yml_path)
        exp = GeneralInferExperiment(experiment_config)

        infer = Inference(exp.model, exp.loaders["test"],
                          **experiment_config["infer_params"])
        out_df = infer.create_sub(sub=exp.sample_sub)
        print(out_df.head())
        print(out_df["EncodedPixels"])
        self.assertTrue(isinstance(out_df, pd.DataFrame))

        # Test how well it actually performed
        make_mask_kwargs = {
            "img_name": self.img_names[0],
            "out_shape_cv2": (525, 350),
            "num_classes": 4
        }
        sub_df = pd.read_csv("submission.csv")
        pred = make_mask(sub_df, shape=(350, 525),
                         **make_mask_kwargs)
        actual = make_mask(self.rle_df, shape=(1400, 2100),
                           **make_mask_kwargs)
        mean_dice = mean_dice_coef(actual[None], pred[None])

        img_name = make_mask_kwargs["img_name"]
        print(f"Mean Dice for {img_name}: {mean_dice}")
        self.assertTrue(mean_dice > 0.5 and mean_dice < 0.9)

        # visual prediction check
        label_names = ["Fish", "Flower", "Gravel", "Sugar"]
        for channel, label in zip(range(4), label_names):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
            ax1.imshow(actual[:, :, channel])
            ax1.set_title(f"{label} Mask")
            ax2.imshow(pred[:, :, channel])
            ax2.set_title(f"{label} Prediction")
            plt.show()

    def test_Inference_create_sub_TTA(self):
        """Testing that Inference.create_sub() runs smoothly with TTA.
        """
        # Seg only for now
        yml_path = "resources/configs/create_sub_tta.yml"
        experiment_config = load_config(yml_path)
        exp = GeneralInferExperiment(experiment_config)

        infer = Inference(exp.model, exp.loaders["test"],
                          **experiment_config["infer_params"])
        out_df = infer.create_sub(sub=exp.sample_sub)
        print(out_df.head())
        print(out_df["EncodedPixels"])
        self.assertTrue(isinstance(out_df, pd.DataFrame))

        # Test how well it actually performed
        make_mask_kwargs = {
            "img_name": self.img_names[0],
            "out_shape_cv2": (525, 350),
            "num_classes": 4
        }
        sub_df = pd.read_csv("submission.csv")
        pred = make_mask(sub_df, shape=(350, 525),
                         **make_mask_kwargs)
        actual = make_mask(self.rle_df, shape=(1400, 2100),
                           **make_mask_kwargs)
        mean_dice = mean_dice_coef(actual[None], pred[None])

        img_name = make_mask_kwargs["img_name"]
        print(f"Mean Dice for {img_name}: {mean_dice}")
        self.assertTrue(mean_dice > 0.5 and mean_dice < 0.9)

        # visual prediction check
        label_names = ["Fish", "Flower", "Gravel", "Sugar"]
        for channel, label in zip(range(4), label_names):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
            ax1.imshow(actual[:, :, channel])
            ax1.set_title(f"{label} Mask")
            ax2.imshow(pred[:, :, channel])
            ax2.set_title(f"{label} Prediction")
            plt.show()


def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true) == 0) and (np.sum(y_pred_bin) == 0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))


def mean_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (n_samples, height, width, n_channels)
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[-1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dice_coef(y_true[i, :, :, j],
                                            y_pred_bin[i, :, :, j])
            mean_dice_channel += channel_dice/(channel_num*batch_size)
    return mean_dice_channel


unittest.main(argv=[''], verbosity=2, exit=False)
