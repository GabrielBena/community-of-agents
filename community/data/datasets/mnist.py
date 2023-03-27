import torch
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST, MNIST
import numpy as np
from typing import Any, Callable, Dict, List, Optional
from typing import Any, AnyStr, Callable, Optional, Tuple
from torchvision import datasets, transforms
from PIL import Image


def estimate_covariance(cov_ratio, n_classes_per_digit, data_size=10000):

    targets = [
        np.random.randint(0, n_classes_per_digit, size=data_size) for _ in range(2)
    ]

    sorted_idxs = np.argsort(targets[0])

    t_idxs = [np.where(targets[1] == t)[0] for t in range(n_classes_per_digit)]
    c_idxs = [np.where(targets[1] != t)[0] for t in range(n_classes_per_digit)]

    idxs = [np.concatenate((t_idx, c_idx)) for t_idx, c_idx in zip(t_idxs, c_idxs)]

    ps = np.stack(
        [
            np.concatenate((np.ones_like(t_idx), cov_ratio * np.ones_like(c_idx)))
            for t_idx, c_idx in zip(t_idxs, c_idxs)
        ],
        -1,
    ).astype(float)
    ps /= ps.sum(0)

    cov_idxs = np.concatenate(
        [
            np.random.choice(idxs[t], size=len(t_idx), p=ps[:, t])
            for t, t_idx in enumerate(t_idxs)
        ]
    )[np.argsort(sorted_idxs)]

    targets[1] = targets[1][cov_idxs]
    targets = np.stack(targets)
    return np.corrcoef(targets)[0, 1]


class Custom_MNIST(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        sorted: bool = False,
        truncate: list or int = None,
        all_digits: bool = False,
    ) -> None:

        self.truncate = np.array(truncate) if truncate is not None else truncate

        super().__init__(root, train, transform, target_transform, download)

        self.sorted = sorted
        self.s_idx = (
            self.targets.sort()[1] if sorted else torch.arange(len(self.targets))
        )
        self.all_digits = all_digits

        if all_digits:
            self.d_idxs = []
            for t in self.truncate_values:
                self.d_idxs.append(torch.where(self.targets == t)[0])

    def _load_data(self):
        data, targets = super()._load_data()
        if self.truncate is not None:
            try:
                truncate_mask = (
                    np.stack([np.array(targets) == t for t in self.truncate])
                    .sum(0)
                    .clip(0, 1)
                    == 1
                )
                mul = 1  # 10//len(self.truncate)
                truncate_values = self.truncate
                self.truncate_values = np.sort(self.truncate)

            except ValueError:
                truncate_mask = np.array(targets) < self.truncate
                print(truncate_mask)
                mul = 1  # 10//self.truncate
                self.truncate_values = np.arange(self.truncate)

            data, targets = torch.cat([data[truncate_mask]] * mul, 0), torch.cat(
                [targets[truncate_mask]] * mul, 0
            )

            for i, t in enumerate(self.truncate_values):
                # targets[targets == t] = i
                """"""
        else:
            self.truncate_values = np.arange(10)

        self.n_classes = len(self.truncate_values)

        return data, targets

    def __getitem__(self, index: int, get_all=None) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if get_all is None:
            get_all = self.all_digits

        if not get_all:
            if self.sorted:
                img, target = self.data[self.s_idx][index], int(
                    self.targets[self.s_idx][index]
                )

            else:
                img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode="L")
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        else:
            # Fetching all digits
            imgs, targets = [], []
            for d_idx in self.d_idxs:
                img, target = self.__getitem__(d_idx[index], False)
                imgs.append(img), targets.append(target)

            # imgs, targets
            return torch.stack(imgs), torch.tensor(targets)

    def __len__(self):
        if not self.all_digits:
            return len(self.targets)
        else:
            return np.min([len(idx) for idx in self.d_idxs])


class Custom_EMNIST(EMNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        data_type: str = "digits",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        truncate: list = None,
    ) -> None:

        self.truncate = truncate
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            split=data_type,
        )

    def _load_data(self):
        data, targets = super()._load_data()
        if self.split == "letters":
            targets -= 1
        self.n_classes = targets.unique().shape[0]
        if self.truncate is not None:
            try:
                truncate_mask = np.array(targets) < self.truncate
                mul = self.n_classes // self.truncate
                truncate_values = np.arange(self.truncate)
            except ValueError:
                truncate_mask = (
                    np.stack([np.array(targets) == t for t in self.truncate])
                    .sum(0)
                    .clip(0, 1)
                    == 1
                )
                # mul = self.n_classes // len(self.truncate)
                mul = 1
                truncate_values = self.truncate
                truncate_values.sort()

            data, targets = torch.cat([data[truncate_mask]] * mul, 0), torch.cat(
                [targets[truncate_mask]] * mul, 0
            )

            for i, t in enumerate(truncate_values):
                targets[targets == t] = i

            self.truncate_values = truncate_values

        self.n_classes = targets.unique().shape[0]

        return data, targets


class DoubleMNIST(Dataset):
    """
    Double Digits MNIST dataset
    Args :
        train : use training set
        asym : solve parity task asymetry by removing digit_1==digit_2 cases
    """

    def __init__(
        self,
        root,
        train=True,
        fix_asym=True,
        permute=False,
        seed=None,
        truncate=None,
        cov_ratio=1,
    ):
        super().__init__()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        dataset = Custom_MNIST(
            root,
            train=train,
            download=True,
            transform=self.transform,
            truncate=truncate,
        )
        self.mnist_dataset = dataset

        if truncate is not None:
            self.n_classes = len(truncate)
        else:
            self.n_classes = 10

        self.fix_asym = fix_asym
        self.permute = permute
        if self.permute:
            if seed is None:
                seed = 42

            torch.manual_seed(seed)
            self.permutation1 = torch.randperm(self.n_classes)
            torch.manual_seed(seed + 1)
            self.permutation2 = torch.randperm(self.n_classes)

            self.permutations = [self.permutation1, self.permutation2]
            # print(self.permutations)
        else:
            self.permutations = [
                torch.arange(self.n_classes),
                torch.arange(self.n_classes),
            ]

        self.cov_ratio = cov_ratio
        self.create_all_idxs()

        self.data = self.create_data()

    def create_all_idxs(self):

        targets = self.mnist_dataset.targets
        sorted_idxs = np.argsort(targets)

        t_idxs = [torch.where(targets == t)[0] for t in range(10)]
        c_idxs = [torch.where(targets != t)[0] for t in range(10)]

        idxs = [np.concatenate((t_idx, c_idx)) for t_idx, c_idx in zip(t_idxs, c_idxs)]

        ps = np.stack(
            [
                np.concatenate(
                    (np.ones_like(t_idx), self.cov_ratio * np.ones_like(c_idx))
                )
                for t_idx, c_idx, idx in zip(t_idxs, c_idxs, idxs)
            ],
            -1,
        ).astype(float)
        ps /= ps.sum(0)

        self.cov_idxs = np.concatenate(
            [
                np.random.choice(idxs[t], size=len(t_idx), p=ps[:, t])
                for t, t_idx in enumerate(t_idxs)
            ]
        )[np.argsort(sorted_idxs)]

        # print((targets == targets[self.cov_idxs]).float().mean())

        if self.fix_asym:
            self.new_idxs = self.get_forbidden_indexs()
        else:
            self.new_idxs = torch.arange(len(self.mnist_dataset))

    def valid_idx(self, idx):
        idx1, idx2 = idx, self.cov_idxs[idx]
        _, target_1 = self.mnist_dataset[idx1]
        _, target_2 = self.mnist_dataset[idx2]

        return not (target_1 == target_2 or target_1 == (target_2 - 1) % 10)

    def get_forbidden_indexs(self):
        new_idxs = []
        for idx in range(len(self.mnist_dataset)):
            if self.valid_idx(idx):
                new_idxs.append(idx)
        return new_idxs

    def __getitem__(self, index, manual_idx=False):

        if manual_idx:

            index_1 = self.new_idxs[index]
            index_2 = self.cov_idxs[index_1]

            digit_1, target_1 = self.mnist_dataset[index_1]
            digit_2, target_2 = self.mnist_dataset[index_2]

            if self.permute:
                target_1, target_2 = (
                    self.permutation1[target_1],
                    self.permutation2[target_2],
                )

            digits = torch.cat([digit_1, digit_2], axis=0)
            targets = torch.tensor([target_1, target_2])
        else:
            digits, targets = [d[index] for d in self.data]

        return digits, targets

    def __len__(self):
        # return len(self.mnist_dataset)
        return len(self.new_idxs)

    def tf_img(self, img):
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def create_data(self):

        self.mnist_data = torch.stack([self.tf_img(d) for d in self.mnist_dataset.data])

        data = [
            [
                d[torch.tensor(idx)]
                for i, d in enumerate([self.mnist_data, self.mnist_dataset.targets])
            ]
            for idx in [self.new_idxs, self.cov_idxs[self.new_idxs]]
        ]

        data = [torch.stack(d, 1).squeeze() for i, d in enumerate(zip(*data))]

        if self.permute:
            data[1] = torch.stack(
                [self.permutations[i][t] for i, t in enumerate(data[1].T)], -1
            )
        data[0] = data[0].float()
        return data


class DoubleDataset(Dataset):
    """
    Double Digits dataset
    Args :
        train : use training set
        asym : solve parity task asymetry by removing digit_1==digit_2 cases
    """

    def __init__(
        self,
        datasets,
        fix_asym=True,
        permute=False,
        seed=None,
        cov_ratio=1,
        transform=None,
    ):
        super().__init__()

        self.transform = (
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            if transform is None
            else transform
        )

        self.datasets = datasets
        assert len(np.unique([d.n_classes for d in datasets])) == 1
        assert len(np.unique([len(d) for d in datasets])) == 1

        self.n_classes = datasets[0].n_classes

        self.fix_asym = fix_asym
        self.permute = permute
        if self.permute:
            if seed is None:
                seed = 42

            torch.manual_seed(seed)
            self.permutation1 = torch.randperm(self.n_classes)
            torch.manual_seed(seed + 1)
            self.permutation2 = torch.randperm(self.n_classes)

            self.permutations = [self.permutation1, self.permutation2]
            # print(self.permutations)
        else:
            self.permutations = [
                torch.arange(self.n_classes),
                torch.arange(self.n_classes),
            ]

        self.cov_ratio = cov_ratio
        self.create_all_idxs()

        self.data = self.create_data()

    def create_all_idxs(self):

        targets = [p[d.targets] for d, p in zip(self.datasets, self.permutations)]

        sorted_idxs = np.argsort(targets[0])

        t_idxs = [torch.where(targets[1] == t)[0] for t in range(self.n_classes)]
        c_idxs = [torch.where(targets[1] != t)[0] for t in range(self.n_classes)]

        idxs = [np.concatenate((t_idx, c_idx)) for t_idx, c_idx in zip(t_idxs, c_idxs)]

        ps = np.stack(
            [
                np.concatenate(
                    (np.ones_like(t_idx), self.cov_ratio * np.ones_like(c_idx))
                )
                for t_idx, c_idx in zip(t_idxs, c_idxs)
            ],
            -1,
        ).astype(float)
        ps /= ps.sum(0)

        self.cov_idxs = np.concatenate(
            [
                np.random.choice(idxs[t], size=len(t_idx), p=ps[:, t])
                for t, t_idx in enumerate(t_idxs)
            ]
        )[np.argsort(sorted_idxs)]

        # print((targets[0] == targets[1][self.cov_idxs]).float().mean())

        if self.fix_asym:
            self.new_idxs = self.get_forbidden_indexs()
        else:
            self.new_idxs = torch.arange(len(self.datasets[0]))

    def valid_idx(self, idx):
        idx1, idx2 = idx, self.cov_idxs[idx]
        _, target_1 = self.datasets[0][idx1]
        _, target_2 = self.datasets[1][idx2]

        return not (target_1 == target_2 or target_1 == (target_2 - 1) % self.n_classes)

    def get_forbidden_indexs(self):
        new_idxs = []
        for idx in range(len(self.datasets[0])):
            if self.valid_idx(idx):
                new_idxs.append(idx)
        return new_idxs

    def __getitem__(self, index, manual_idx=False):

        if manual_idx:

            index_1 = self.new_idxs[index]
            index_2 = self.cov_idxs[index_1]

            digit_1, target_1 = self.datasets[0][index_1]
            digit_2, target_2 = self.datasets[1][index_2]

            if self.permute:
                target_1, target_2 = (
                    self.permutation1[target_1],
                    self.permutation2[target_2],
                )

            digits = torch.cat([digit_1, digit_2], axis=0)
            targets = torch.tensor([target_1, target_2])
        else:

            digits, targets = [d[index] for d in self.data]

        return digits, targets

    def __len__(self):
        # return len(self.mnist_dataset)
        return len(self.new_idxs)

    def tf_img(self, img):
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def create_data(self):

        self.img_data = [
            torch.stack([self.tf_img(d) for d in dataset.data])
            for dataset in self.datasets
        ]

        data = [
            [d[torch.tensor(idx)] for i, d in enumerate([img_data, dataset.targets])]
            for idx, img_data, dataset in zip(
                [self.new_idxs, self.cov_idxs[self.new_idxs]],
                self.img_data,
                self.datasets,
            )
        ]

        data = [torch.stack(d, 1).squeeze() for i, d in enumerate(zip(*data))]

        if self.permute:
            data[1] = torch.stack(
                [self.permutations[i][t] for i, t in enumerate(data[1].T)], -1
            )
        data[0] = data[0].float()
        return data


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
