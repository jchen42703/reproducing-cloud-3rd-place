import numpy as np
import os
import cv2

from torch.utils.data import Dataset


class CloudDataset(Dataset):
    def __init__(self, img_ids: np.array, data_folder: str, masks_folder: str,
                 transforms=None, preprocessing=None):
        """
        Attributes
            img_ids (np.ndarray): of image names.
            data_folder (str): path to the image directory
            masks_folder (str): path to the masks directory
            transforms (albumentations.augmentation): transforms to apply
                before preprocessing. Defaults to None.
            preprocessing (albumentations.augmentation): ops to perform after
                transforms, such as z-score standardization. Defaults to None.
        """
        self.data_folder = data_folder
        self.masks_folder = masks_folder

        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]

        # loading image
        img_path = os.path.join(self.data_folder, img_name)
        img = cv2.cvtColor(np.load(img_path), cv2.COLOR_BGR2RGB)
        # loading mask
        mask_path = os.path.join(self.masks_folder, img_name)
        mask = np.load(mask_path).astype(np.float32)

        # apply augmentations
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed["image"]
            mask = preprocessed["mask"]
        return img, mask

    def __len__(self):
        return len(self.img_ids)
