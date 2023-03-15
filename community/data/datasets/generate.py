import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST, Omniglot
import torchvision.transforms.functional as tF
import numpy as np
from community.data.datasets.mnist import Custom_EMNIST, DoubleMNIST, Custom_MNIST
from community.data.datasets.symbols import SymbolsDataset


class MultiDataset(Dataset):
    def __init__(self, datasets, shuffle=False, fix_asym=False):

        self.datasets = datasets
        self.small_dataset_idx = np.argmin([len(d) for d in self.datasets])
        self.small_dataset = datasets[self.small_dataset_idx]
        self.shuffle = shuffle

        self.fix_asym = fix_asym
        self.secondary_idxs = [
            torch.randperm(len(self.small_dataset))
            if i != self.small_dataset_idx and shuffle
            else torch.arange(len(self.small_dataset))
            for i, d in enumerate(self.datasets)
        ]

        self.n_classes = len(self.small_dataset.targets.unique())

        if self.fix_asym:
            self.new_idxs = self.get_forbidden_indexs()
        else:
            self.new_idxs = torch.arange(len(self.small_dataset))

    def valid_idx(self, idx):
        idx1, idx2 = self.secondary_idxs[0][idx], self.secondary_idxs[1][idx]
        _, target_1 = self.datasets[0][idx1]
        _, target_2 = self.datasets[1][idx2]

        valid = not (
            target_1 == (target_2 - 1) % self.n_classes or target_1 == target_2
        )

        return valid

    def get_forbidden_indexs(self):
        new_idxs = []
        for idx in range(len(self.small_dataset)):
            if self.valid_idx(idx):
                new_idxs.append(idx)
        return new_idxs

    def __len__(self):
        # return  len(self.small_dataset)
        return len(self.new_idxs)

    def __getitem__(self, idx):
        # get images and labels here
        # returned images must be tensor
        # labels should be int
        idx = self.new_idxs[idx]
        idxs = [s_idx[idx] for s_idx in self.secondary_idxs]

        samples = [d[idx] for (d, idx) in zip(self.datasets, idxs)]
        datas, labels = [d[0] for d in samples], [d[1] for d in samples]
        try:
            return torch.stack(datas), torch.tensor(labels)
        except:
            return datas, labels


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
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    truncate_digits = np.arange(n_classes)

    kwargs = train_kwargs, test_kwargs

    single_digits = [
        Custom_MNIST(
            root, train=t, download=True, transform=transform, truncate=truncate_digits
        )
        for t in [True, False]
    ]

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

    single_fashion = [
        FashionMNIST(root, train=t, transform=transform, download=True)
        for t in [True, False]
    ]

    transform = transforms.Compose(
        [
            lambda img: transforms.functional.rotate(img, -90),
            lambda img: transforms.functional.hflip(img),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

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
                root, train=t, data_type="balanced", truncate=trunc, transform=transform
            )
            for t in [True, False]
        ]
        for trunc in truncates
    ]

    multi_datasets = [
        MultiDataset([s1, s2], fix_asym=False)
        for (s1, s2) in zip(single_digits, single_letters[0])
    ]
    double_letters = [
        MultiDataset([s1, s2], fix_asym=False, shuffle=True)
        for (s1, s2) in zip(single_letters[0], single_letters[1])
    ]

    datasets = [
        single_digits,
        double_digits,
        double_letters,
        multi_datasets,
        single_letters,
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

    return loaders


def get_datasets_symbols(data_config, use_cuda=True, n_classes=10, plot=False):

    symbol_config = data_config["symbol_config"]
    batch_size = data_config["batch_size"]
    symbol_config["common_input"] = data_config["common_input"]

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
