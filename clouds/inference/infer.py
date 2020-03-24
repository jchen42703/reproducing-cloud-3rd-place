import cv2
import tqdm
import os
from functools import partial

from clouds.inference.utils import mask2rle, post_process, \
                                   apply_nonlin


class Inference(object):
    """General Inference class.

    Methods:
        create_sub: Creates the submission file
        get_encoded_pixels: Gets the segmentation encoded pixels
    """
    def __init__(self, model, test_loader, tta_flips=None, class_params=None):
        """
        Args:
            model (torch.nn.Module): callable object that returns the
                forward pass of a model
                To ensemble, see `experiments.utils.load_checkpoints` and
                `experiments.utils.EnsembleModel`.
            test_loader (torch.utils.data.DataLoader): Test loader
            tta_flips (list-like): consisting one of or all of
                ["lr_flip", "ud_flip", "lrud_flip"]. Defaults to None.
            class_params (dict): of {class: {threshold, min_size}}
                min_size is the minimum size a segmentation can be to be
                considered as "correct"
        """
        self.model = model
        self.loader = test_loader

        if class_params is None:
            # class: (threshold, min_size)
            self.class_params = {0: (0.5, 10000), 1: (0.5, 10000),
                                 2: (0.5, 10000), 3: (0.5, 10000)}
        else:
            self.class_params = class_params

        self.tta_fn = None
        if tta_flips is not None:
            assert isinstance(tta_flips, (list, tuple)), \
                "tta_flips must be a list-like of strings."
            print(f"TTA Ops: {tta_flips}")
            self.tta_fn = partial(tta_flips_fn, model=self.model,
                                  mode="segmentation", flips=tta_flips,
                                  non_lin="sigmoid")

    def get_encoded_pixels(self):
        """
        Processes predicted logits and converts them to encoded pixels. Does
        so in an iterative manner so operations are done image-wise rather than
        on the full dataset directly (to combat RAM limitations).

        Returns:
            encoded_pixels: list of rles in the order of self.loader
        """
        encoded_pixels = []
        image_id = 0
        for test_x in tqdm.tqdm(self.loader):
            if self.tta_fn is not None:
                pred_out = self.tta_fn(batch=test_x.cuda())
            else:
                pred_out = apply_nonlin(self.model(test_x.cuda()))
            # for each batch (4, h, w): resize and post_process
            for i, batch in enumerate(pred_out):
                for prob in batch:
                    # iterating through each probability map (h, w)
                    prob = prob.cpu().detach().numpy()
                    if prob.shape != (350, 525):
                        # cv2 -> (w, h); np -> (h, w)
                        prob = cv2.resize(prob, dsize=(525, 350),
                                          interpolation=cv2.INTER_LINEAR)
                    thresh = self.class_params[image_id % 4][0]
                    min_size = self.class_params[image_id % 4][1]
                    predict, num_predict = post_process(prob,
                                                        thresh,
                                                        min_size,
                                                        pred_shape=(350, 525))
                    if num_predict == 0:
                        encoded_pixels.append("")
                    else:
                        r = mask2rle(predict)
                        encoded_pixels.append(r)
                    image_id += 1
        return encoded_pixels

    def create_sub(self, sub):
        """
        Creates and saves a submission dataframe (classification/segmentation).
        Args:
            sub (pd.DataFrame): the same sub used for the test dataset;
                the sample_submission dataframe (stage1). This is used to
                create the final submission dataframe
        Returns:
            submission (pd.DataFrame): submission dataframe
        """
        print("Segmentation: Converting predicted masks to",
              "run-length-encodings...")
        save_path = os.path.join(os.getcwd(), "submission.csv")
        encoded_pixels = self.get_encoded_pixels()

        # Saving the submission dataframe
        sub["EncodedPixels"] = encoded_pixels
        sub.fillna("")
        sub.to_csv(save_path, columns=["Image_Label", "EncodedPixels"],
                   index=False)
        print(f"Saved the submission file at {save_path}")
        return sub
