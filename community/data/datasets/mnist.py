import torch
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST, MNIST
import numpy as np
from typing import Any, Callable, Dict, List, Optional
from typing import Any, AnyStr, Callable, Optional, Tuple
from torchvision import datasets, transforms
from PIL import Image


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

        self.truncate = np.array(truncate)
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
                mul = self.n_classes // len(self.truncate)
                truncate_values = self.truncate
                truncate_values.sort()

            data, targets = torch.cat([data[truncate_mask]] * mul, 0), torch.cat(
                [targets[truncate_mask]] * mul, 0
            )

            for i, t in enumerate(truncate_values):
                targets[targets == t] = i

            self.truncate_values = truncate_values

        return data, targets


class DoubleMNIST(Dataset):
    """
    Double Digits MNIST dataset
    Args :
        train : use training set
        asym : solve parity task asymetry by removing digit_1==digit_2 cases
    """

    def __init__(
        self, root, train=True, fix_asym=True, permute=False, seed=None, truncate=None
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
        self.secondary_index = torch.randperm(len(self.mnist_dataset))
        if self.fix_asym:
            self.new_idxs = self.get_forbidden_indexs()
            # self.secondary_index = self.secondary_index[: len(self.new_idxs)]
        else:
            self.new_idxs = torch.arange(len(dataset))

        self.permute = permute
        if self.permute:
            if seed is not None:
                torch.manual_seed(seed)
            self.permutation = torch.randperm(self.n_classes)
            # print(self.permutation)

    def valid_idx(self, idx):
        idx1, idx2 = idx, self.secondary_index[idx]
        _, target_1 = self.mnist_dataset[idx1]
        _, target_2 = self.mnist_dataset[idx2]

        return not (target_1 == target_2 or target_1 == (target_2 - 1) % 10)

    def permute_labels(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.permutation = torch.randperm(10)

    def get_forbidden_indexs(self):
        new_idxs = []
        for idx in range(len(self.mnist_dataset)):
            if self.valid_idx(idx):
                new_idxs.append(idx)
        return new_idxs

    def __getitem__(self, index):
        index_1 = self.new_idxs[index]
        index_2 = self.secondary_index[index_1]

        digit_1, target_1 = self.mnist_dataset[index_1]
        digit_2, target_2 = self.mnist_dataset[index_2]

        if self.permute:
            target_1, target_2 = self.permutation[target_1], self.permutation[target_2]

        digits = torch.cat([digit_1, digit_2], axis=0)
        targets = [target_1, target_2]

        return digits, torch.tensor(targets)

    def __len__(self):
        # return len(self.mnist_dataset)
        return len(self.new_idxs)

    @property
    def data(self):
        datas = [
            [
                d[torch.tensor(idx)]
                for d in [self.mnist_dataset.data, self.mnist_dataset.targets]
            ]
            for idx in [self.new_idxs, self.secondary_index[self.new_idxs]]
        ]
        return [torch.stack(d, 1) for i, d in enumerate(zip(*datas))]
