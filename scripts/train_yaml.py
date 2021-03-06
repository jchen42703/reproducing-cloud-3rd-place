from catalyst.dl.runner import SupervisedRunner

from clouds.experiments import TrainSegExperiment, seed_everything


def main(config):
    """Main code for training a classification model.

    Args:
        config (dict): dictionary read from a yaml file
            i.e. configs/train_seg1.yml

    Returns:
        None

    """
    # setting up the train/val split with filenames
    seed = config["io_params"]["split_seed"]
    seed_everything(seed)

    # Seg only for now
    exp = TrainSegExperiment(config)
    output_key = "logits"

    print(f"Seed: {seed}")

    runner = SupervisedRunner(output_key=output_key)

    runner.train(
        model=exp.model,
        criterion=exp.criterion,
        optimizer=exp.opt,
        scheduler=exp.lr_scheduler,
        loaders=exp.loaders,
        callbacks=exp.cb_list,
        logdir=config["runner_params"]["logdir"],
        num_epochs=config["runner_params"]["num_epochs"],
        valid_loader="val",
        verbose=config["runner_params"]["verbose"],
        fp16=config["runner_params"]["fp16"]
    )


if __name__ == "__main__":
    import argparse
    from clouds.experiments.utils import load_config

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    config = load_config(args.yml_path)
    main(config)
