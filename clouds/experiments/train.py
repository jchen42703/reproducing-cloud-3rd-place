from abc import abstractmethod
import catalyst.dl.callbacks as callbacks
from catalyst.utils import get_device, any2device
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss

from clouds.losses import *
from clouds.custom_lr_schedulers import WarmRestartsCustomScheduler
from .utils import load_weights

torch.optim.lr_scheduler.__dict__.update({
    "WarmRestartsCustomScheduler": WarmRestartsCustomScheduler
})


class TrainExperiment(object):
    """Base Training Experiment Class

    This class holds all the components necessary for running a training
    experiment, such as:
    - model
    - datasets/loaders (ids for splits)
    - callbacks
    - optimizer
    - learning rate scheduler
    - criterion
    It relies on configs, which are located in the `configs` folder.

    """
    def __init__(self, config: dict):
        # for reuse
        self.config = config
        self.cb_params = config["callback_params"]
        self.criterion_params = config["criterion_params"]
        self.io_params = config["io_params"]
        self.model_params = config["model_params"]
        self.opt_params = config["opt_params"]
        # initializing the experiment components
        self.df = self.setup_df()
        self.train_ids, self.val_ids, = self.get_split()
        self.train_dset, self.val_dset = self.get_datasets(self.train_ids,
                                                           self.val_ids)
        self.loaders = self.get_loaders()
        self.model = self.get_model()
        self.opt = self.get_opt()
        self.lr_scheduler = self.get_lr_scheduler()
        self.criterion = self.get_criterion()
        self.cb_list = self.get_callbacks()

    @abstractmethod
    def get_datasets(self, train_ids, val_ids):
        """Initializes transforms and datasets.

        Initializes the data augmentation and preprocessing transforms. Creates
        and returns the train and validation datasets.
        """
        return

    @abstractmethod
    def get_model(self):
        """Creates and returns the model.
        """
        return

    def setup_df(self):
        """Setting up the dataframe to have the `img_id` & `label` columns;

        Meanings:
            img_id: the base img name (without the class name)
            label: the label name

        Returns:
            train (pd.DataFrame): dataframe with the columns:
                - Image_Label
                - EncodedPixels
                - img_id
                - label

        """
        train_csv_path = self.config["train_csv_path"]
        train = pd.read_csv(train_csv_path)
        train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
        train["img_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])
        return train

    def get_split(self):
        """Creates train/val filename splits

        Returns:
            (train_ids, val_ids)

        """
        # setting up the train/val split with filenames
        split_seed: int = self.io_params["split_seed"]
        test_size: float = self.io_params["test_size"]
        # doing the splits
        print("Splitting the df normally...")
        img_ids = self.df["img_id"].drop_duplicates().values
        train_ids, val_ids = train_test_split(img_ids,
                                              random_state=split_seed,
                                              test_size=test_size)
        return (train_ids, val_ids)

    def get_loaders(self):
        """Creates train/val loaders from datasets

        The datasets are created in self.get_datasets.

        Returns:
            dictionary with keys:
            - 'train': train loader
            - 'val': validation loader

        """
        # setting up the loaders
        batch_size = self.io_params["batch_size"]
        num_workers = self.io_params["num_workers"]
        train_loader = DataLoader(self.train_dset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(self.val_dset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

        self.train_steps = len(self.train_dset)  # for schedulers
        return {"train": train_loader, "val": val_loader}

    def get_opt(self):
        """Creates the optimizer

        """
        assert isinstance(self.model, torch.nn.Module), \
            "`model` must be an instance of torch.nn.Module`"
        # fetching optimizers
        opt_name = self.opt_params["opt"]
        opt_kwargs = self.opt_params[opt_name]
        opt_cls = torch.optim.__dict__[opt_name]
        opt = opt_cls(filter(lambda p: p.requires_grad,
                             self.model.parameters()),
                      **opt_kwargs)
        print(f"Optimizer: {opt}")
        return opt

    def get_lr_scheduler(self):
        """Creates the LR scheduler from `self.opt`

        The optimizer should be created in `self.get_opt`.

        Returns:
            schedulder (torch.optim.lr_scheduler):

        """
        assert isinstance(self.opt, torch.optim.Optimizer), \
            "`optimizer` must be an instance of torch.optim.Optimizer"
        sched_params = self.opt_params["scheduler_params"]
        scheduler_name = sched_params["scheduler"]
        if scheduler_name is not None:
            scheduler_args = sched_params[scheduler_name]
            scheduler_cls = torch.optim.lr_scheduler.__dict__[scheduler_name]
            scheduler = scheduler_cls(optimizer=self.opt, **scheduler_args)
            print(f"LR Scheduler: {scheduler.__class__.__name__}")
        else:
            scheduler = None
            print("No LR Scheduler")
        return scheduler

    def get_criterion(self):
        """Fetches the criterion. (Only one loss.)

        Returns:
            loss (torch.nn.Module):

        """
        loss_name = self.criterion_params["loss"]
        loss_kwargs = self.criterion_params[loss_name]
        if "weight" in list(loss_kwargs.keys()):
            if isinstance(loss_kwargs["weight"], list):
                weight_tensor = torch.tensor(loss_kwargs["weight"])
                weight_tensor = any2device(weight_tensor, get_device())
                print(f"Converted the `weight` argument in {loss_name}",
                      f" to a {weight_tensor.type()}...")
                loss_kwargs["weight"] = weight_tensor
        loss_cls = globals()[loss_name]
        loss = loss_cls(**loss_kwargs)
        print(f"Criterion: {loss}")
        return loss

    def get_callbacks(self):
        """Creates a list of callbacks.

        Loads each callback with the specified kwargs and also loads weights
        automatically with `CheckpointCallback` if `checkpoint_path` is
        given.

        Returns:
            cb_list (List[catalyst.dl.callbacks]):

        """
        cb_name_list = list(self.cb_params.keys())
        cb_name_list.remove("checkpoint_params")
        cb_list = [callbacks.__dict__[cb_name](**self.cb_params[cb_name])
                   for cb_name in cb_name_list]
        cb_list = self.load_weights(cb_list)
        print(f"Callbacks: {[cb.__class__.__name__ for cb in cb_list]}")
        return cb_list

    def load_weights(self, callbacks_list):
        """Loads model weights with catalyst.dl.callbacks.CheckpointCallback

        Loads model weights and appends the CheckpointCallback if doing
        stateful model loading. This doesn't add the CheckpointCallback if
        it's 'model_only' loading bc SupervisedRunner adds it by default.

        Args:
            callbacks_list (List[catalyst.dl.callbacks]):

        Returns:
            callbacks_list (List[catalyst.dl.callbacks]): the argument list,
                but with the CheckpointCallback if mode == 'full'

        """
        ckpoint_params = self.cb_params["checkpoint_params"]
        # Having checkpoint_params=None is a hacky way to say no checkpoint
        # callback but eh what the heck
        if ckpoint_params["checkpoint_path"] is not None:
            mode = ckpoint_params["mode"].lower()
            if mode == "full":
                print("Stateful loading...")
                ckpoint_p = Path(ckpoint_params["checkpoint_path"])
                fname = ckpoint_p.name
                # everything in the path besides the base file name
                resume_dir = str(ckpoint_p.parents[0])
                print(f"Loading {fname} from {resume_dir}. \
                      \nCheckpoints will also be saved in {resume_dir}.")
                # adding the checkpoint callback
                ckpoint = [callbacks.CheckpointCallback(resume=fname,
                                                        resume_dir=resume_dir)]
                callbacks_list = callbacks_list + ckpoint
            elif mode == "model_only":
                print("Loading weights into model...")
                ckpoint_path = ckpoint_params["checkpoint_path"]
                self.model = load_weights(ckpoint_path, self.model).train()
        return callbacks_list
