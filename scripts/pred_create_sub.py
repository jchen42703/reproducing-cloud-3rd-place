import gc
import torch

from clouds.inference import Inference
from clouds.experiments import GeneralInferExperiment


def main(config):
    """For creating the segmentation-only submission file.
    All masks are converted to either "" or RLEs

    Args:
        config (dict): dictionary read from a yaml file
            i.e. config/

    Returns:
        None
    """
    torch.cuda.empty_cache()
    gc.collect()

    exp = GeneralInferExperiment(config)
    infer = Inference(exp.loaders["test"], models=exp.models,
                      **config["infer_params"])
    out_df = infer.create_sub(sub=exp.sample_sub)


if __name__ == "__main__":
    import argparse
    from clouds.experiments.utils import load_config

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    config = load_config(args.yml_path)
    main(config)

