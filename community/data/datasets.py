import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import EMNIST
import numpy as np
import numpy.random as rd
from typing import Any, Callable, Dict, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fmin_cobyla
from typing import Any, AnyStr, Callable, Optional, Tuple
from torchvision.datasets import MNIST
from PIL import Image
from itertools import permutations


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

        self.truncate = truncate
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
        if self.truncate:
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


class Custom_EMNIST(datasets.EMNIST):
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

    def __init__(self, root, train=True, fix_asym=True, permute=False, seed=None):
        super().__init__()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        dataset = datasets.MNIST(
            root, train=train, download=True, transform=self.transform
        )
        self.mnist_dataset = dataset

        self.fix_asym = fix_asym
        self.secondary_index = torch.randperm(len(self.mnist_dataset))
        if self.fix_asym:
            self.new_idxs = self.get_forbidden_indexs()
        else:
            self.new_idxs = [i for i in range(len(dataset))]

        self.permute = permute
        if self.permute:
            if seed is not None:
                torch.manual_seed(seed)
            self.permutation = torch.randperm(10)
            print(self.permutation)

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


class SymbolsDataset(Dataset):
    def __init__(self, data_config, plot=False) -> None:
        super().__init__()

        self.data_config = data_config

        self.symbols = self.get_symbols()
        self.symbol_size = self.symbols[0].shape[0]
        self.n_symbols = data_config["n_symbols"]
        self.double_data = data_config["double_data"]

        if plot:
            fig, axs = plt.subplots(1, len(self.symbols))
            for ax, sym in zip(axs, self.symbols):
                ax.imshow(sym)
            plt.show()

        self.fixed_symbol_number = False

        # Set this to True to regenerate random symbol assignments at each call of __getitem__
        self.regenerate = False

        self.data = self.generate_data()

    def get_symbols(self):

        s_type = self.data_config["symbol_type"]
        n_diff = self.data_config["n_diff_symbols"]

        if s_type == "0":

            symbols = [np.zeros((5, 5)) for n in range(2)]

            for i in range(5):
                for j in range(5):

                    if (i + j) % 4 == 2 or (i - j) % 4 == 2:
                        symbols[1][i, j] = 1

                    if i % 4 == 0 or j % 4 == 0:
                        symbols[0][i, j] = 1

        elif s_type == "1":

            X = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
            C = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

            symbols = [X, C]

        elif "random" in s_type:
            s_size = int(s_type.split("_")[-1]) - 1
            symbols = [
                np.random.randint(0, 2, size=(s_size, s_size)) for n in range(n_diff)
            ]

        elif "mod" in s_type:
            s_size = int(s_type.split("_")[-1])
            symbols = [np.zeros((s_size + 1, s_size + 1)) for n in range(n_diff)]
            step = s_size // n_diff

            # self.data_config["symbol_size"] += 1

            for i in range(s_size):
                for j in range(s_size):
                    for n in range(n_diff):

                        if (
                            (i + j) % (s_size - 1) == n or (i - j) % (s_size - 1) == n
                        ) and n % 2 == 0:
                            symbols[n][i, j] = 1

                        if (
                            i % (s_size - 1) == n - 1 or j % (s_size - 1) == n - 1
                        ) and n % 2 == 1:
                            symbols[n][i, j] = 1

        else:
            raise NotImplementedError(
                ' Provide symbol type in [0, 1, "random_{size}", "mod_{size}"] '
            )

        return symbols

    def get_probabilities(self, n_classes):

        get_tns = lambda x: np.array(
            [(x**2).sum()]
            + [2 * pn * (x[: n + 1].sum()) for n, pn in enumerate(x[1:])]
        )
        min_tns = lambda tns: ((tns - (np.ones_like(tns) / len(tns))) ** 2).sum()
        min_f = lambda x: min_tns(get_tns(x))

        constraints = []

        f_cons_0 = lambda x: np.ones(n_classes).dot(x) - 1

        # cons_0 = LinearConstraint(np.ones(n_classes), 1, 1)
        cons_0 = {"type": "eq", "fun": f_cons_0}

        """
        for k in range(n_classes) : 
            A = np.zeros(n_classes)
            A[k] = 1
            print(A)
            constraints.append(LinearConstraint(A, 0, np.inf))

        """

        x_init = np.ones(n_classes) / n_classes

        x_final = fmin_cobyla(min_f, x_init, (f_cons_0))

        if not x_final.sum() == 1:
            x_final += np.ones(n_classes) * (1 - x_final.sum()) / n_classes

        return x_final

    def get_symbol_positions(
        self, data_size, nb_steps, n_symbols, symbol_size, input_size, static
    ):

        assert np.remainder(input_size, symbol_size) == 0
        n_grid = input_size // symbol_size

        assert n_symbols <= n_grid**2

        if static:
            squares = list(range((n_grid) ** 2))
            positions = np.stack(
                [
                    np.random.choice(squares, n_symbols, replace=False)
                    for _ in range(data_size)
                ]
            )
            centers = np.stack(
                [
                    np.array(np.unravel_index(positions, (n_grid, n_grid))).transpose(
                        1, 2, 0
                    )
                    * symbol_size
                    for _ in range(nb_steps)
                ]
            )
            jitter = np.repeat(
                np.random.random_integers(-1, 0, (data_size, 2))[:, None, :],
                n_symbols,
                axis=1,
            )

        else:
            centers = np.stack(
                [
                    [
                        np.stack(
                            self.get_random_trajectory(
                                nb_steps, input_size, symbol_size
                            )
                        ).T
                        for _ in range(n_symbols)
                    ]
                    for _ in range(data_size)
                ]
            ).transpose(2, 0, 1, -1)

        if not self.data_config["static"]:
            centers = np.concatenate((centers, centers[::-1]))

        return centers

    def get_random_symbol_assignement(self, label):
        assignments = np.zeros(self.n_symbols, dtype=int)

        try:
            assignments[:label] = 1
            if not self.fixed_symbol_number:
                assignments[label:] = np.random.random_integers(
                    -1, 0, self.n_symbols - label
                )
        except TypeError:

            # assign 1st label
            assignments[: label[0]] = 0
            for l in range(1, len(label)):

                # assign other labels
                assignments[label[:l].sum() : label[: l + 1].sum()] = l

            # remaining are set at -1 (no symbol)
            assignments[label.sum() :] = -1

        np.random.shuffle(assignments)
        # print(assignments)
        return assignments

    def place_symbols_from_centers(
        self,
        centers,
        labels,
        data_size,
        input_size,
        symbol_size,
        inv=False,
        symbol_assigns=None,
    ):

        symbols = self.symbols

        if inv:
            symbols = self.symbols[::-1]

        if (not self.fixed_symbol_number) or (not self.double_data):
            symbols.append(np.zeros_like(self.symbols[0]))

        grids = []

        def assign_square(grid, center_pos, l, d):
            grid[
                d,
                center_pos[0] : center_pos[0] + symbol_size,
                center_pos[1] : center_pos[1] + symbol_size,
            ] += symbols[l]

        if symbol_assigns is None or None in symbol_assigns:
            # symbol_assigns = [np.random.choice(self.symbol_assignments_len[l]) for l in labels]
            # symbol_assignments = [self.symbol_assignments[l][n_assign] for l, n_assign in zip(labels, symbol_assigns)]
            symbol_assignments = [self.get_random_symbol_assignement(l) for l in labels]
        else:
            symbol_assignments = symbol_assigns

        for t, center in enumerate(centers):
            grid = np.zeros((data_size, input_size, input_size))
            for d in range(data_size):
                for i, c in enumerate(center[d]):
                    # l = int(i < labels[d])
                    assign_square(grid, (c[0], c[1]), symbol_assignments[d][i], d)

            grids.append(grid)

        return np.stack(grids)

    def get_random_trajectory(
        self, seq_length, image_size, symbol_size, step_length=0.2
    ):

        """
        Generate a trajectory
        https://tcapelle.github.io/pytorch/fastai/cv/2021/05/01/moving_mnist.html
        """
        canvas_size = image_size - symbol_size
        x, y = np.random.random(2)
        v_min = 1 / (step_length * canvas_size)
        v_x, v_y = np.random.random(2) * (1 - v_min) + v_min

        assert v_x * step_length * canvas_size >= 1
        assert v_y * step_length * canvas_size >= 1

        out_x, out_y = [], []

        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * step_length
            x += v_x * step_length

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            out_x.append(x * canvas_size)
            out_y.append(y * canvas_size)

        return torch.tensor(out_x).int(), torch.tensor(out_y).int()

    def get_symbol_data(
        self,
        data_size,
        nb_steps,
        n_symbols,
        symbol_type,
        input_size,
        static,
        double_data,
        n_diff_symbols,
    ):

        symbol_size = self.symbol_size

        # probas = self.get_probabilities(n_symbols+1)

        if not self.double_data:
            probas = np.ones((n_symbols + 1) // n_diff_symbols) / (
                (n_symbols + 1) // n_diff_symbols
            )
            labels = (
                np.stack(
                    [
                        np.random.multinomial(1, probas, size=(data_size)).argmax(-1)
                        for _ in range(n_diff_symbols)
                    ],
                    -1,
                )
                + 1
            )
            centers = self.get_symbol_positions(
                data_size, nb_steps, n_symbols, symbol_size, input_size, static
            )
            grids = self.place_symbols_from_centers(
                centers, labels, data_size, input_size, symbol_size
            )

            return (
                torch.from_numpy(grids).transpose(0, 1),
                torch.from_numpy(labels),
                torch.from_numpy(centers).transpose(0, 1),
            )  # , torch.from_numpy(jitter)

        else:
            probas = np.ones((n_symbols + 1) // n_diff_symbols) / (
                (n_symbols + 1) // n_diff_symbols
            )
            labels = (
                np.stack(
                    [
                        np.random.multinomial(1, probas, size=(data_size)).argmax(-1)
                        for _ in range(2)
                    ],
                    -1,
                )
                + 1
            )
            centers = np.stack(
                [
                    self.get_symbol_positions(
                        data_size, nb_steps, n_symbols, symbol_size, input_size, static
                    )
                    for _ in range(2)
                ]
            )
            grids = np.stack(
                [
                    self.place_symbols_from_centers(
                        c, l, data_size, input_size, symbol_size, inv
                    )
                    for l, c, inv in zip(labels.T, centers, [True, False])
                ]
            )

            return (
                torch.from_numpy(grids).transpose(0, 2),
                torch.from_numpy(labels),
                torch.from_numpy(centers).transpose(0, 2),
            )  # , torch.from_numpy(jitter)

    def generate_data(self):

        data = self.get_symbol_data(**self.data_config)

        return data

    def __len__(self):
        return self.data_config["data_size"]

    def __getitem__(self, index: Any, inv=False, symbol_assigns=[None, None]):

        if not self.regenerate:
            return self.data[0][index], self.data[1][index] - 1
        else:
            centers, labels = self.data[-1][index].unsqueeze(2), self.data[1][index]
            new_data = []
            for ag, (c, l, i, s_a) in enumerate(
                zip(centers, labels, [inv, not inv], symbol_assigns)
            ):
                new_data.append(
                    self.place_symbols_from_centers(
                        c,
                        [l],
                        1,
                        self.data_config["input_size"],
                        self.symbol_size,
                        inv=i,
                        symbol_assigns=[s_a],
                    )[:, 0, ...]
                )

            return torch.from_numpy(np.stack(new_data)).transpose(0, 1), labels - 1


def get_datasets_alphabet(
    root,
    batch_size=256,
    use_cuda=True,
    fix_asym=False,
    n_classes=10,
    split_classes=True,
):

    train_kwargs = {"batch_size": batch_size, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "shuffle": False, "drop_last": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 0, "pin_memory": True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    kwargs = train_kwargs, test_kwargs

    single_digits = [
        datasets.MNIST(root, train=t, download=True, transform=transform)
        for t in [True, False]
    ]
    double_digits = [DoubleMNIST(root, train=t, fix_asym=False) for t in [True, False]]

    single_fashion = [
        datasets.FashionMNIST(root, train=t, transform=transform, download=True)
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
        MultiDataset([s1, s2], fix_asym=fix_asym, shuffle=True)
        for (s1, s2) in zip(single_letters[0], single_letters[1])
    ]

    single_loaders_dig = [
        torch.utils.data.DataLoader(d, **k) for d, k in zip(single_digits, kwargs)
    ]
    double_loaders_dig = [
        torch.utils.data.DataLoader(d, **k) for d, k in zip(double_digits, kwargs)
    ]
    double_loaders_letters = [
        torch.utils.data.DataLoader(d, **k) for d, k in zip(double_letters, kwargs)
    ]
    multi_loaders = [
        torch.utils.data.DataLoader(d, **k) for d, k in zip(multi_datasets, kwargs)
    ]

    return (
        multi_loaders,
        double_loaders_dig,
        double_loaders_letters,
        single_loaders_dig,
        single_letters,
    )


def get_datasets_symbols(
    data_config, batch_size=128, use_cuda=True, n_classes=10, plot=False
):

    train_kwargs = {"batch_size": batch_size, "shuffle": True, "drop_last": True}
    test_kwargs = {"batch_size": batch_size, "shuffle": False, "drop_last": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 0, "pin_memory": True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    kwargs = train_kwargs, test_kwargs

    data_configs = (
        {k: v[0] if i == 0 else v for i, (k, v) in enumerate(data_config.items())},
        {k: v[1] if i == 0 else v for i, (k, v) in enumerate(data_config.items())},
    )
    datasets = [SymbolsDataset(d, plot=plot) for d in data_configs]
    # datasets[1].symbols = datasets[0].symbols
    dataloaders = [
        torch.utils.data.DataLoader(d, **k) for d, k in zip(datasets, kwargs)
    ]

    return dataloaders, datasets
