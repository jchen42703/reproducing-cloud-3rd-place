import os
import cv2
from tqdm import tqdm
from pathlib import Path
from glob import glob

import numpy as np

from clouds.io.utils import make_mask_single

COLAB_PATHS_DICT = {
    "train_dir": "./train_images/",
    "test_dir": "./test_images/",
    "train_out": "train640",
    "test_out": "test640",
    "masks_out": "masks640",
}


class Preprocessor(object):
    """Preprocessor class.

    Attributes:
        df: dataframe with cols ["Image_Label", "EncodedPixels"];
            the first dataframe from running `setup_train_and_sub_df(...)`
        train_dir (str / None):
        test_dir (str / None):
        train_out (str / None):
        test_out (str / None):
        masks_out (str / None):
            Leave as None, if not using a specific route.
        out_shape_cv2 (tuple): (w, h); reverse of numpy shaping (how
            cv2 handles its input sizing)
        train_fpaths (List[str]):
        test_fpaths (List[str]):

    """
    def __init__(self, df, paths_dict=COLAB_PATHS_DICT,
                 out_shape_cv2=(640, 320)):
        """

        Args:
            df: dataframe with cols ["Image_Label", "EncodedPixels"];
                the first dataframe from running `setup_train_and_sub_df(...)`
            paths_dict (dict): for all of the paths to the input and output
                directories and files.
                Keys:
                - train_dir
                - test_dir
                - train_out: path to the output training images zip
                - test_out: path to the output test images zip
                - masks_out
                Leave as None, if not using a specific route.
            out_shape_cv2 (tuple): (w, h); reverse of numpy shaping (how
                cv2 handles its input sizing)

        """
        self.df = df
        # parsing the paths_dict dictionary
        # setting default values (in the event that the user is missing keys)
        keys_list = ["train_dir", "test_dir", "train_out", "test_out",
                     "masks_out"]
        for key in keys_list:
            setattr(self, key, None)
        # setting actual values from the dict
        for key in paths_dict.keys():
            setattr(self, key, paths_dict[key])

        # Gathering the file path lists
        if self.train_dir is not None:
            assert os.path.isdir(self.train_dir), \
                "Please make sure train_dir is a directory."
            self.train_fpaths = glob(os.path.join(self.train_dir, "*.jpg"),
                                     recursive=True)
            print(f"{len(self.train_fpaths)} training images")

        if self.test_dir is not None:
            assert os.path.isdir(self.test_dir), \
                "Please make sure test_dir is a directory."
            self.test_fpaths = glob(os.path.join(self.test_dir, "*.jpg"),
                                    recursive=True)
            print(f"{len(self.test_fpaths)} test images")

        self.out_shape_cv2 = out_shape_cv2

    def execute_images(self, out_dir, img_fpaths):
        """Preprocesses masks and saves them in .npy files.

        Creates the masks from rles in the provided dataframe, resizes them,
        and saves them as numpy arrays.

        Args:
            out_dir (str): Path to the output directory.
                This will generally be one of the attributes created from
                `paths_dict`.
            img_fpaths (List[str]): List of all the image paths to read from
                This is generally done through glob.glob().

        Returns:
            None

        """
        for img_fpath in tqdm(img_fpaths, total=len(img_fpaths)):
            img = read_and_resize_img(img_fpath,
                                      out_shape_cv2=self.out_shape_cv2)
            save_path = os.path.join(out_dir, f"{Path(img_fpath).stem}.npy")
            np.save(save_path, img)

    def execute_masks(self):
        """Creates and resizes the masks, which are saved in .npy files.

        Creates the masks from rles in the provided dataframe, resizes them,
        and saves them as numpy arrays.

        """
        all_img_ids = self.df["Image_Label"].apply(lambda x: x.split("_")[0])
        all_img_ids = all_img_ids.drop_duplicates().values

        for img_name in tqdm(all_img_ids):
            mask = make_mask(self.df, img_name, shape=(1400, 2100),
                             out_shape_cv2=self.out_shape_cv2, num_classes=4)
            save_path = os.path.join(self.masks_out,
                                     f"{Path(img_name).stem}.npy")
            np.save(save_path, mask)

    def execute_all(self):
        """General method for preprocessing everything.

        Runs self.execute_images for the training/testing images and
        self.execute_masks for the masks. If a directory is not specified,
        it will be skipped.

        """
        if self.train_out is not None and self.train_dir is not None:
            print("\nPreprocessing training images...")
            assert self.train_fpaths is not None, \
                "Make sure that train_dir is specified."
            self.execute_images(self.train_out, self.train_fpaths)

        if self.test_out is not None and self.test_dir is not None:
            print("\nPreprocessing test images...")
            assert self.test_fpaths is not None, \
                "Make sure that test_dir is specified."
            self.execute_images(self.test_out, self.test_fpaths)

        if self.masks_out is not None:
            print("\nPreprocessing masks...")
            self.execute_masks()


def read_and_resize_img(img_fpath, out_shape_cv2=(640, 320)):
    """Reads an image and resizes it.

    Args:
        img_fpath (str): path to an image
            i.e. /content/00b81e1.jpg
        out_shape_cv2 (Tuple[int]): shape to resize to without channels and in
        cv2 format.

    Returns:
        img (np.ndarray): resized array

    """
    img = np.array(cv2.imread(img_fpath))
    img = cv2.resize(img, out_shape_cv2)
    return img


def make_mask(df, img_name, shape=(1400, 2100), out_shape_cv2=(576, 384),
              num_classes=4):
    """Creates masks from rles, resizes them, and combines them into an array.

    Args:
        df (pd.DataFrame): dataframe from train.csv
        img_name (str): image name (without the class)
            i.e. 00b81e1.jpg
        shape (Tuple[int]): initial shape of the mask without channels (h, w)
            This is the shape that the mask is when it is read from an RLE.
        out_shape_cv2 (Tuple[int]): shape to resize to without channels and in
            cv2 format.
        num_classes (int): number of classes

    Returns:
        mask (np.ndarray): binary array with type int and shape:
            (`shape`, num_classes)

    """
    mask = np.zeros((out_shape_cv2[1], out_shape_cv2[0], num_classes))
    for label_idx, label in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
        rle_mask = make_mask_single(df, label, img_name,
                                    shape=shape)
        # 3rd place interpolates with bilinear and then thresholds
        resized = (cv2.resize(rle_mask, out_shape_cv2) > 0).astype(int)
        mask[:, :, label_idx] = resized
    return mask.astype(np.uint8)
