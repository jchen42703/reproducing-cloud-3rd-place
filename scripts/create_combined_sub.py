import pandas as pd

from clouds.inference.cascade import combine_stage1_stage2


def main(config):
    stage1_df = pd.read_csv(config["stage1_csv_path"])
    stage2_df = pd.read_csv(config["stage2_csv_path"])
    combined = combine_stage1_stage2(stage1_df, stage2_df)
    combined.to_csv(config["save_path"])


if __name__ == "__main__":
    import argparse
    from clouds.experiments.utils import load_config

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    config = load_config(args.yml_path)
    main(config)
