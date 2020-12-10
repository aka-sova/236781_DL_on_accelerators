import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cs236781.train_results import FitResult

from . import cnn, training

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = dict(
    cnn=cnn.ConvClassifier, resnet=cnn.ResNetClassifier, ycn=cnn.YourCodeNet
)


def run_experiment(
    run_name,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=None,
    lr=1e-3,
    reg=1e-3,
    # Model params
    filters_per_layer=[64],
    layers_per_block=2,
    pool_every=2,
    hidden_dims=[1024],
    model_type="cnn",
    # You can add extra configuration for your experiments here
    pooling_params=dict(kernel_size=2),
    conv_params=dict(kernel_size=3, stride=1, padding=1),
    activation_type='relu',
    activation_params={},
    pooling_type="max",
    **kw):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======

    # 0. create dataloaders
    dl_train = torch.utils.data.DataLoader(ds_train, bs_train, shuffle=False)
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False)

    # 1. create model based on parameters.
    #       Calculate the amount of blocks/layers if that's resnet or simple cnn

    # duplicate each filter by layers per block
    channels = [filt for filt in filters_per_layer for block_layer in range(layers_per_block)]
    # print(f"channels = {channels}")
    in_size = ds_train.data[0].shape # H,W,C
    in_size = tuple(in_size[i] for i in [2, 0, 1])

    # find target space
    out_classes = len(set(ds_train.targets))

    if model_type == "cnn":
        model = cnn.ConvClassifier(in_size=in_size,
                                   out_classes=out_classes,
                                   channels=channels,
                                   pool_every=pool_every,
                                   hidden_dims=hidden_dims,
                                   pooling_params=pooling_params,
                                   conv_params=conv_params,
                                   activation_type = activation_type,
                                   activation_params = activation_params,
                                   pooling_type = pooling_type,
                                   **kw)
    else:
        model = cnn.ResNetClassifier(in_size=in_size,
                                     out_classes=out_classes,
                                     channels=channels,
                                     pool_every=pool_every,
                                     hidden_dims=hidden_dims,
                                     pooling_params=pooling_params,
                                     conv_params=conv_params,
                                     activation_type=activation_type,
                                     activation_params=activation_params,
                                     pooling_type=pooling_type,
                                     **kw)

    # 2. create optimizer. Weight decay is L2 penalty
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg)

    # 3. create loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # 4. create TorchTrainer with model, optimizer, loss function, device
    trainer = training.TorchTrainer(model=model,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    device=device)

    # 5. train the Trainer for desired number of epochs, get fit_res
    #       num_batches is calculates inside trainer: num_batches =  dataloader / batchsize
    #       if this number exceeds max_batches, then it is equal max_batches
    fit_res = trainer.fit(dl_train=dl_train,
                          dl_test=dl_test,
                          num_epochs=epochs,
                          checkpoints=checkpoints,
                          early_stopping=early_stopping,
                          max_batches=batches,
                          **kw)
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS236781 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=run_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
