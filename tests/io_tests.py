import unittest

import cv2
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from clouds.io.utils import make_mask_single, rle_decode


class PreprocessingTests(unittest.TestCase):
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

        self.out_shape_cv2 = (576, 384)

    def test_make_mask_single(self):
        """Tests `make_mask_single`
        """
        # creating + resizing
        label_list = ["Fish", "Flower", "Gravel", "Sugar"]
        for img_name in self.img_names:
            for classidx, class_name in enumerate(label_list):
                mask = make_mask_single(self.rle_df, class_name, img_name,
                                        shape=(1400, 2100))
                self.assertTrue(isinstance(mask, np.ndarray))

    def test_rle_decoding(self):
        """Comparing my rle decode with the 3rd place's solution.
        """
        label_list = ["Fish", "Flower", "Gravel", "Sugar"]
        for img_name in self.img_names:
            for class_name in label_list:
                img_label = f"{img_name}_{class_name}"
                shape = (1400, 2100)

                condition = self.rle_df["Image_Label"] == img_label
                encoded = self.rle_df.loc[condition,
                                          "EncodedPixels"].values
                # handling NaNs and longer rles
                encoded = encoded[0] if len(encoded) == 1 else encoded
                mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
                mask_3rd = np.zeros((shape[0], shape[1]), dtype=np.float32)
                if encoded is not np.nan:
                    mask = rle_decode(encoded, shape=shape)
                    mask_3rd = rle2mask_3rd_place(encoded, height=1400,
                                                  width=2100)
                self.assertTrue(np.array_equal(mask, mask_3rd))

    def test_make_mask(self):
        """
        def make_mask(df, img_name, shape=(1400, 2100),
                      out_shape_cv2=(576, 384), num_classes=4):
        """
        pass

    def test_mask_resizing(self):
        """Testing mask resizing interpolation differences.

        Testing to see if resizing the masks with INTER_NEAREST v.
        INTER_LINEAR matters. More explicitly, it tests that the amount of
        disagreement is < 1% of the total number of pixels.

        for label in ["Fish", "Flower", "Gravel", "Sugar"]:
            mask = make_mask_single(self.df, label, img_name,
                                    shape=(1400, 2100))*255
            mask = cv2.resize(mask, self.out_shape_cv2,
                              interpolation=cv2.INTER_NEAREST)
        v.
            mask = rle2mask(rle, height=1400, width=2100, fill_value=1)
            mask = (cv2.resize(mask, (W, H)) > 0).astype(int)
            new_rle = mask2rle(mask)

        """
        # creating the mask to be resized
        # (384, 576) masks
        mask_shape = (self.out_shape_cv2[1], self.out_shape_cv2[0], 4)
        nearest_mask = np.zeros(mask_shape, dtype=np.float32)
        bilinear_mask = np.zeros(mask_shape, dtype=np.float32)

        # creating + resizing
        label_list = ["Fish", "Flower", "Gravel", "Sugar"]
        for img_name in self.img_names:
            for classidx, class_name in enumerate(label_list):
                mask = make_mask_single(self.rle_df, class_name, img_name,
                                        shape=(1400, 2100))
                resized_nearest = cv2.resize(mask,
                                             self.out_shape_cv2,
                                             interpolation=cv2.INTER_NEAREST)
                nearest_mask[:, :, classidx] = resized_nearest
                resized_bilinear = cv2.resize(mask,
                                              self.out_shape_cv2,
                                              interpolation=cv2.INTER_LINEAR)
                bilinear_mask[:, :, classidx] = resized_bilinear

        # thresholding
        nearest_mask = (nearest_mask > 0).astype(int)
        bilinear_mask = (bilinear_mask > 0).astype(int)
        # overlay masks and see the number of points that have = 1 vote
        overlap = nearest_mask + bilinear_mask
        num_disagreements = len(np.where(overlap == 1))

        nearest_distr = np.unique(nearest_mask, return_counts=True)
        bilinear_distr = np.unique(bilinear_mask, return_counts=True)

        print(f"Nearest Distr: {nearest_distr}",
              f"\nBilinear_Distr: {bilinear_distr}")
        print(f"{num_disagreements} mask disagreements.")
        # testing to see if the num_disagreements is less than 1% of the total
        # number of pixels
        self.assertTrue(num_disagreements < overlap.flatten().size * 0.01)


def rle2mask_3rd_place(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height, width), np.float32)
    if rle != '':
        mask = mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start, length in r:
            start = start - 1  # ???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask = mask.reshape(width, height).T
    return mask


unittest.main(argv=[''], verbosity=2, exit=False)
