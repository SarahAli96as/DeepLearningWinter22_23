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

from .cnn import CNN, ResNet
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
}


def mlp_experiment(
    depth: int,
    width: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    dl_test: DataLoader,
    n_epochs: int,
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======

        
    model = BinaryClassifier(
        model= MLP(
            in_dim=len(dl_train.dataset.tensors[0][0]),
            dims=[*[width]*depth, 2],
            nonlins=[*['tanh', ]*depth, 'softmax'] 
        ),
        threshold=0.5,
    )
    
    ## Custom paramters
    lr = 0.3
    weight_decay = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    momentum = 0.7
    
    ## Aggregate into dict
    custom_parameters = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    
    optimizer = torch.optim.SGD(params=model.parameters(), **custom_parameters)
    trainer = ClassifierTrainer(model, loss_fn, optimizer)

    fit_result = trainer.fit(dl_train, dl_valid, num_epochs=n_epochs, print_every=0);
    
    thresh = select_roc_thresh(model, *dl_valid.dataset.tensors, plot=False)
    
    valid_acc = fit_result.test_acc[-1]
    
    trainer.model.threshold = thresh 
    
    test_acc = trainer.test_epoch(dl_test, verbose = False).accuracy
    
    # ========================
    return model, thresh, valid_acc, test_acc


def cnn_experiment(
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
    **kw,
):
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
    # ------------------------------------
    # --------- Model Definition ---------
    # ------------------------------------
    ## Majd: Will try to use a model as close as possible to ResNet50 based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    ## Default conv layers have kernel=3 with pad=1 to preserve size.
    ## In ResNet50, pooling is done only at the end before the FC layers - Need to validate this and try different values. Avg Pool was used with 4 kernel size for (3, 64, 64) input image. Will use 2 kernel size since out input is (3,32,32).
    x0, _ = ds_train[0]
    in_size = list(x0.shape)
    out_classes = len(ds_train.classes)
    
    ## Define Channels
    channels = [val for val in filters_per_layer for _ in range(layers_per_block)]
    pool_every = len(channels)
    
    ## Parse **kw for common Resnet/CNN parameters
    res_cnn_parameters = {}
    
    if 'conv_params' in kw:
        res_cnn_parameters['conv_params']= kw['conv_params']
    else:
        res_cnn_parameters['conv_params']= {'kernel_size': 3, 'padding': 1, 'stride': 1}
        
    if 'activation_type' in kw:
        res_cnn_parameters['activation_type']= kw['activation_type']
        
    if 'activation_params' in kw:
        res_cnn_parameters['activation_params']= kw['activation_params']
        
    if 'pooling_type' in kw:
        res_cnn_parameters['pooling_type']= kw['pooling_type']
    else:
        res_cnn_parameters['pooling_type']= 'avg'
        
    if 'pooling_params' in kw:
        res_cnn_parameters['pooling_params']= kw['pooling_params']
    else:
        res_cnn_parameters['pooling_params']= {'kernel_size': 2}
        
    ## Parse **kw for Resnet-only parameters
    res_only_parameters = {}
    
    if 'batchnorm' in kw:
        res_only_parameters['batchnorm']= kw['batchnorm']
    else:
        res_only_parameters['batchnorm']= True
    if 'dropout' in kw:
        res_only_parameters['dropout']= kw['dropout']
    if 'bottleneck' in kw:
        res_only_parameters['bottleneck']= kw['bottleneck']
    else:
        res_only_parameters['bottleneck']= True
    
    
    
    if model_cls is CNN:
        model = ArgMaxClassifier(model_cls(in_size=in_size, out_classes=out_classes,
        channels=channels, pool_every=pool_every, hidden_dims=hidden_dims, **res_cnn_parameters))
    else:
        model = ArgMaxClassifier(model_cls(in_size=in_size, out_classes=out_classes,
        channels=channels, pool_every=pool_every, hidden_dims=hidden_dims, **res_cnn_parameters, **res_only_parameters))  
    
    # --------------------------------
    # --------- Data Loaders ---------
    # --------------------------------
    train_size = batches * bs_train
    dl_train = torch.utils.data.DataLoader(ds_train, bs_train,
                                       sampler=torch.utils.data.SubsetRandomSampler(range(0,train_size)))
    print(f'Train size is: {train_size}')
    
    
    test_size = min(batches * bs_test, 6000)
    print(f'Test size is: {test_size}')
    
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test,
                                       sampler=torch.utils.data.SubsetRandomSampler(range(0,test_size)))
       
    # -----------------------------------------------
    # --------- Optimizer and Loss Function ---------
    # -----------------------------------------------
    ## CrossEntropy Loss function 
    loss_fn = torch.nn.CrossEntropyLoss()
    
    ## Optimizer Parameters
    optimizer_parameters = dict(lr=lr, weight_decay = reg)
        
    if 'momentum' in kw:
        optimizer_parameters['momentum']= kw['momentum']
    else: 
        optimizer_parameters['momentum']= 0.9
    
    optimizer = torch.optim.SGD(params=model.parameters(), **optimizer_parameters)
    
    # ----------------------------
    # --------- Training ---------
    # ----------------------------  
    trainer = ClassifierTrainer(model, loss_fn, optimizer, device = device)

    fit_res = trainer.fit(dl_train, dl_test, num_epochs=epochs, print_every=0, checkpoints=checkpoints, early_stopping=early_stopping)
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
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
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
