import pandas as pd

from clouds.preprocess import Preprocessor


def main(config):
    paths_params = config["paths_params"]
    paths_dict = {
        "train_dir": paths_params["train_dir"],
        "test_dir": paths_params["test_dir"],
        "train_out": paths_params["train_out"],
        "test_out": paths_params["test_out"],
        "masks_out": paths_params["masks_out"],
    }

    train = pd.read_csv(paths_params["train_csv_path"])
    preprocessor = Preprocessor(train, paths_dict,
                                tuple(config["out_shape_cv2"]))
    preprocessor.execute_all()


if __name__ == "__main__":
    import argparse
    from clouds.experiments.utils import load_config

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    config = load_config(args.yml_path)
    main(config)
