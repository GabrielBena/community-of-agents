import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cobyla
from typing import Any, AnyStr, Callable, Optional, Tuple
from torchvision.transforms import functional as F, InterpolationMode


class SymbolsDataset(Dataset):
    def __init__(self, data_config, plot=False) -> None:
        super().__init__()

        self.data_config = data_config
        self.random_transform = data_config["random_transform"]
        self.max_angle = 15

        self.symbols = self.get_symbols()
        self.symbol_size = self.symbols[0].shape[-1]
        self.n_symbols = data_config["n_symbols"]
        self.common_input = data_config["common_input"]
        self.cov_ratio = data_config["cov_ratio"]

        if plot:
            if self.random_transform:
                fig, axs = plt.subplots(1, len(self.symbols))
                for ax, sym in zip(axs, self.symbols):
                    ax.imshow(
                        np.concatenate(
                            [sym[0], sym[2 * self.max_angle // 2], sym[-1]], -1
                        )
                    )
                plt.show()
            else:

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

            symbol_orders = np.array([0, 3, 2, 1])
            if len(symbol_orders) < n_diff:
                symbol_orders = np.concatenate(
                    (symbol_orders, np.arange(len(symbol_orders), n_diff))
                )

            # self.data_config["symbol_size"] += 1

            for i in range(s_size):
                for j in range(s_size):
                    for n, (n_mod, _) in enumerate(zip(symbol_orders, range(n_diff))):

                        if (
                            (i + j) % (s_size - 1) == n_mod
                            or (i - j) % (s_size - 1) == n_mod
                        ) and n_mod % 2 == 0:

                            symbols[n][i, j] = 1

                        if (
                            i % (s_size - 1) == n_mod - 1
                            or j % (s_size - 1) == n_mod - 1
                        ) and n_mod % 2 == 1:

                            symbols[n][i, j] = 1

        else:
            raise NotImplementedError(
                ' Provide symbol type in [0, 1, "random_{size}", "mod_{size}"] '
            )

        if self.random_transform:
            symbols = [
                np.stack(
                    [
                        F.rotate(
                            torch.tensor(sym).unsqueeze(0),
                            angle=i,
                            interpolation=InterpolationMode.BILINEAR,
                        )
                        .data.squeeze()
                        .numpy()
                        for i in range(-self.max_angle, self.max_angle)
                    ],
                    0,
                )
                for sym in symbols
            ]

        return symbols

    def get_probabilities(self, n_classes):

        get_tns = lambda x: np.array(
            [(x**2).sum()]
            + [2 * pn * (x[: n + 1].sum()) for n, pn in enumerate(x[1:])]
        )
        min_tns = lambda tns: ((tns - (np.ones_like(tns) / len(tns))) ** 2).sum()
        min_f = lambda x: min_tns(get_tns(x))

        f_cons_0 = lambda x: np.ones(n_classes).dot(x) - 1

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

            def get_all_trajectories():
                return [
                    np.stack(
                        self.get_random_trajectory(nb_steps, input_size, symbol_size)
                    ).T
                    for _ in range(data_size)
                ]

            if self.data_config["parallel"]:

                centers = np.stack(
                    ray.get([get_all_trajectories.remote() for s in range(n_symbols)]),
                    1,
                ).transpose(
                    2, 0, 1, -1
                )  # timesteps x data_size x n_symbols x 2

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
                ).transpose(
                    2, 0, 1, -1
                )  # timesteps x data_size x n_symbols x 2

        # if not self.data_config["static"]:
        # centers = np.concatenate((centers, centers[::-1]))

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

        if (not self.fixed_symbol_number) or (self.common_input):
            symbols.append(np.zeros_like(self.symbols[0]))

        n_steps, n_symbols = centers.shape[0], centers.shape[-2]
        grids = np.zeros((n_steps, data_size, input_size, input_size))

        if symbol_assigns is None or None in symbol_assigns:
            # symbol_assigns = [np.random.choice(self.symbol_assignments_len[l]) for l in labels]
            # symbol_assignments = [self.symbol_assignments[l][n_assign] for l, n_assign in zip(labels, symbol_assigns)]
            symbol_assignments = np.stack(
                [self.get_random_symbol_assignement(l) for l in labels]
            )
        else:
            symbol_assignments = symbol_assigns

        if self.data_config["parallel"]:

            grids = np.zeros((n_steps, data_size, input_size, input_size))
            grids = ray.put(grids)

            # centers_img = ray.put(centers)
            # symbols_img = ray.put(symbol_assignments)

            @ray.remote
            def fill_grid(time_step, data_idx, symbol):
                center_pos = centers[time_step, data_idx, symbol]
                label = symbol_assignments[data_idx][symbol]
                grids[
                    data_idx,
                    center_pos[0] : center_pos[0] + symbol_size,
                    center_pos[1] : center_pos[1] + symbol_size,
                ] += symbols[label]

            @ray.remote
            def fill_grid_per_ts(time_step):

                grids = np.zeros((data_size, input_size, input_size))
                for data_idx in range(data_size):
                    for symbol in range(n_symbols):
                        center_pos = centers[time_step, data_idx, symbol]
                        label = symbol_assignments[data_idx][symbol]
                        grids[
                            data_idx,
                            center_pos[0] : center_pos[0] + symbol_size,
                            center_pos[1] : center_pos[1] + symbol_size,
                        ] += symbols[label]

                return grids

            @ray.remote
            def fill_grid_per_data(data_idx):
                grids = np.zeros((n_steps, input_size, input_size))
                for time_step in range(n_steps):
                    for symbol in range(n_symbols):
                        center_pos = centers[time_step, data_idx, symbol]
                        label = symbol_assignments[data_idx][symbol]
                        grids[
                            time_step,
                            center_pos[0] : center_pos[0] + symbol_size,
                            center_pos[1] : center_pos[1] + symbol_size,
                        ] += symbols[label]

                return grids

            @ray.remote
            def fill_grid_per_symbol(symbol):
                grids = np.zeros((n_steps, data_size, input_size, input_size))
                for time_step in range(n_steps):
                    for data_idx in range(data_size):
                        center_pos = centers[time_step, data_idx, symbol]
                        label = symbol_assignments[data_idx][symbol]
                        grids[
                            time_step,
                            data_idx,
                            center_pos[0] : center_pos[0] + symbol_size,
                            center_pos[1] : center_pos[1] + symbol_size,
                        ] += symbols[label]

                return grids

            if True:

                grids = np.stack(
                    ray.get([fill_grid_per_ts.remote(idx) for idx in range(n_steps)])
                )

            else:

                grids = np.stack(
                    ray.get(
                        [fill_grid_per_symbol.remote(idx) for idx in range(n_symbols)]
                    )
                )

                grids = np.stack(
                    ray.get(
                        [fill_grid_per_data.remote(idx) for idx in range(data_size)]
                    ),
                    1,
                )

            ray.shutdown()

        else:

            grids = []

            def assign_square(grid, center_pos, l, d):

                sym = symbols[l].copy()
                if self.random_transform:
                    sym = sym[np.random.randint(-self.max_angle, self.max_angle)]
                grid[
                    d,
                    center_pos[0] : center_pos[0] + symbol_size,
                    center_pos[1] : center_pos[1] + symbol_size,
                ] += sym

            if symbol_assigns is None or None in symbol_assigns:
                # symbol_assigns = [np.random.choice(self.symbol_assignments_len[l]) for l in labels]
                # symbol_assignments = [self.symbol_assignments[l][n_assign] for l, n_assign in zip(labels, symbol_assigns)]
                symbol_assignments = [
                    self.get_random_symbol_assignement(l) for l in labels
                ]
            else:
                symbol_assignments = symbol_assigns

            for center in centers:
                grid = np.zeros((data_size, input_size, input_size))
                for d in range(data_size):
                    for i, c in enumerate(center[d]):
                        # l = int(i < labels[d])
                        assign_square(grid, (c[0], c[1]), symbol_assignments[d][i], d)

                grids.append(grid)

            grids = np.stack(grids)

        return grids

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
        input_size,
        static,
        n_diff_symbols,
        **others,
    ):

        symbol_size = self.symbol_size

        if self.data_config["adjust_probas"]:
            self.probas = self.get_probabilities((n_symbols + 1) // n_diff_symbols)
        else:
            self.probas = np.ones((n_symbols + 1) // n_diff_symbols) / (
                (n_symbols + 1) // n_diff_symbols
            )

        if self.cov_ratio == 1:
            labels = (
                np.stack(
                    [
                        np.random.multinomial(1, self.probas, size=(data_size)).argmax(
                            -1
                        )
                        for _ in range(n_diff_symbols)
                    ],
                    -1,
                )
                + 1
            )
        else:
            n_labels = len(self.probas)

            cov_matrix = np.ones((n_labels, n_labels))
            cov_matrix[~np.eye(n_labels, dtype=bool)] *= self.cov_ratio
            if self.cov_ratio > 0:
                cov_matrix /= cov_matrix.sum(0)

            first_labels = np.random.multinomial(
                1, self.probas, size=(data_size)
            ).argmax(-1)

            cov_probas = cov_matrix[first_labels]
            multi_sample = lambda p: np.random.multinomial(1, p, size=1).argmax(-1)
            labels = [
                np.stack([multi_sample(p) for p in cov_probas]).squeeze()
                for _ in range(n_diff_symbols - 1)
            ]
            labels.insert(0, first_labels)
            labels = np.stack(labels, -1) + 1

        if self.common_input:

            centers = self.get_symbol_positions(
                data_size, nb_steps, n_symbols, symbol_size, input_size, static
            )
            grids = self.place_symbols_from_centers(
                centers, labels, data_size, input_size, symbol_size
            )

            return (
                torch.from_numpy(grids).transpose(0, 1),
                torch.from_numpy(labels) - 1,
                torch.from_numpy(centers).transpose(0, 1),
            )  # , torch.from_numpy(jitter)

        else:

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
                torch.from_numpy(labels) - 1,
                torch.from_numpy(centers).transpose(0, 2),
            )  # , torch.from_numpy(jitter)

    def generate_data(self):

        data = self.get_symbol_data(**self.data_config)

        return data

    def __len__(self):
        return self.data_config["data_size"]

    def __getitem__(self, index: Any, inv=False, symbol_assigns=[None, None]):

        if not self.regenerate:
            return self.data[0][index], self.data[1][index]
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

            return torch.from_numpy(np.stack(new_data)).transpose(0, 1), labels
