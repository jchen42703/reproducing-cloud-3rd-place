from google_drive_downloader import GoogleDriveDownloader as gdd
import torch


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
    weights_id = "1-2ON7yRtpYX62yvnZ7tCb-WJs0_mOUqL"
    save_path = "./sample_submission.csv"
    gdd.download_file_from_google_drive(file_id=weights_id,
                                        dest_path=save_path,
                                        unzip=False)


def download_weights():
    """Downloads example weights for tests.
    """
    # 0.346 dice on 00a0954.jpg
    # weights_id = "1YMewsRkJoybsy4Qs05UJG_0ZvgL8pZg5"
    # save_path = "./fpn_resnet34_seg1_seed350_mvp_best.pth"
    # 0.71 dice on 00a0954.jpg
    weights_id = "1ibc0aNyQxxNvPqix9CABAKAAH5p6iL4d"
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
