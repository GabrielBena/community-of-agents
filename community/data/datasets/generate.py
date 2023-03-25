import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST, Omniglot
import torchvision.transforms.functional as tF
import numpy as np
from community.data.datasets.mnist import (
    Custom_EMNIST,
    DoubleDataset,
    Custom_MNIST,
)
from community.data.datasets.symbols import SymbolsDataset


def get_datasets_alphabet(root, data_config):

    batch_size = data_config["batch_size"]
    data_sizes = data_config["data_size"]
    use_cuda = data_config["use_cuda"]
    fix_asym = data_config["fix_asym"]
    n_classes = data_config["n_classes_per_digit"]
    split_classes = data_config["split_classes"]
    permute = data_config["permute_dataset"]
    seed = data_config["seed"]
    cov_ratio = data_config["cov_ratio"]

    train_kwargs = {"batch_size": batch_size, "shuffle": True, "drop_last": True}
    test_kwargs = {"batch_size": batch_size, "shuffle": False, "drop_last": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4, "pin_memory": True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform_digits = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    transform_letters = transforms.Compose(
        [
            lambda img: transforms.functional.rotate(img, -90),
            lambda img: transforms.functional.hflip(img),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    truncate_digits = np.arange(n_classes)

    kwargs = train_kwargs, test_kwargs

    single_digits = [
        Custom_MNIST(
            root,
            train=t,
            download=True,
            transform=transform_digits,
            truncate=truncate_digits,
        )
        for t in [True, False]
    ]

    single_fashion = [
        FashionMNIST(root, train=t, transform=transform_digits, download=True)
        for t in [True, False]
    ]

    truncates = np.arange(10, 47)
    excludes = [18, 19, 21]  # exclude I, J, L
    for e in excludes:
        truncates = truncates[truncates != e]
    # np.random.shuffle(truncates)
    if split_classes:
        assert n_classes <= 17, "Max 17 classes for a separate set for each agents"
        truncates = truncates[: n_classes * 2]
        truncates = np.split(truncates, 2)
    else:
        assert n_classes <= 34, "Max 34 classes for the same set for each agents"
        truncates = truncates[:n_classes]
        truncates = truncates, truncates

    single_letters = [
        [
            Custom_EMNIST(
                root,
                train=t,
                data_type="balanced",
                truncate=trunc,
                transform=transform_letters,
            )
            for t in [True, False]
        ]
        for trunc in truncates
    ]

    """
    double_digits = [
        DoubleMNIST(
            root,
            train=t,
            fix_asym=fix_asym,
            truncate=truncate_digits,
            permute=permute,
            seed=seed,
            cov_ratio=cov_ratio,
        )
        for t in [True, False]
    ]


    multi_datasets = [
            MultiDataset([s1, s2], fix_asym=False)
            for (s1, s2) in zip(single_digits, single_letters[0])
        ]
        double_letters = [
            MultiDataset([s1, s2], fix_asym=False, shuffle=True)
            for (s1, s2) in zip(single_letters[0], single_letters[1])
        ]

    """

    double_digits = [
        DoubleDataset(
            [d, d],
            fix_asym=fix_asym,
            permute=permute,
            seed=seed,
            cov_ratio=cov_ratio,
            transform=transform_digits,
        )
        for d in single_digits
    ]

    double_letters = [
        DoubleDataset(
            [s1, s2],
            fix_asym=fix_asym,
            permute=permute,
            seed=seed,
            cov_ratio=cov_ratio,
            transform=transform_letters,
        )
        for (s1, s2) in zip(single_letters[0], single_letters[1])
    ]

    datasets = [
        single_digits,
        double_digits,
        single_letters,
        double_letters,
    ]
    if data_sizes is not None:
        datasets = [
            [
                torch.utils.data.Subset(d, torch.arange(d_size))
                for d, d_size in zip(dsets, data_sizes)
            ]
            for dsets in datasets
        ]

    loaders = [
        [torch.utils.data.DataLoader(d, **k) for d, k in zip(dsets, kwargs)]
        for dsets in datasets
    ]

    return {
        k: [dataset, loader]
        for k, dataset, loader in zip(
            [
                "single_digits",
                "double_digits",
                "single_letters",
                "double_letters",
            ],
            datasets,
            loaders,
        )
    }


def get_datasets_symbols(data_config, use_cuda=True, n_classes=10, plot=False):

    symbol_config = data_config["symbol_config"].copy()
    batch_size = data_config["batch_size"]
    symbol_config["common_input"] = data_config["common_input"]

    symbol_config["cov_ratio"] = data_config["cov_ratio"]
    symbol_config["nb_steps"] = data_config["nb_steps"]
    symbol_config["n_diff_symbols"] = data_config["n_digits"]
    symbol_config["n_symbols"] = data_config["n_classes"] - 1

    train_kwargs = {"batch_size": batch_size, "shuffle": True, "drop_last": True}
    test_kwargs = {"batch_size": batch_size, "shuffle": False, "drop_last": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4, "pin_memory": True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    kwargs = train_kwargs, test_kwargs

    data_configs = (
        {
            k: v[0] if k == "data_size" else v
            for i, (k, v) in enumerate(symbol_config.items())
        },
        {
            k: v[1] if k == "data_size" else v
            for i, (k, v) in enumerate(symbol_config.items())
        },
    )
    datasets = [SymbolsDataset(d, plot=plot) for d in data_configs]
    # datasets[1].symbols = datasets[0].symbols
    dataloaders = [
        torch.utils.data.DataLoader(d, **k) for d, k in zip(datasets, kwargs)
    ]

    return dataloaders, datasets


def get_omni_dataset(root="../../data/", size=16):
    transform = transforms.Compose(
        [
            lambda img: tF.resize(img, [size, size]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            lambda tensor: -tensor,
        ]
    )

    omni = Omniglot(root, transform=transform, background=False)

    return omni
