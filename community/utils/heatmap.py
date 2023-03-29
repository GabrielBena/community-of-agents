import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata as gd


def gaussian_filter(x, y, sigmas):

    try:
        sigmas[0]
    except (TypeError, IndexError) as e:
        sigmas = [sigmas, sigmas]

    eps = 1e-10
    filter = np.exp(-((x / sigmas[0]) ** 2 + (y / sigmas[1]) ** 2))

    return filter


def filter_nans(values):
    idxs = np.isnan(values[-1])
    return [v[~idxs] for v in values]


def weighted_average(x, y, sigmas, values):

    x_values, y_values, z_values = values

    return (
        (z_values * gaussian_filter(x - x_values, y - y_values, sigmas))
    ).sum() / gaussian_filter(x - x_values, y - y_values, sigmas).sum()


def plot_filters(smoothness, values):
    x_values, y_values, _ = values

    Y = np.linspace(y_values.min(), y_values.max(), 100)
    X = np.linspace(x_values.min(), x_values.max(), 100)

    ratio = X / Y
    sigmas = np.stack([ratio, np.ones_like(ratio)]) * smoothness
    points = [(p1, p2) for p1 in [10, 90] for p2 in [10, 90]]

    fig, axs = plt.subplots(1, len(points), figsize=(3 * len(points), 3))

    for (p1, p2), ax in zip(points, axs):

        point = [X[p1], Y[p2]]
        filter = lambda x, y: gaussian_filter(
            (x - point[0]), y - point[1], sigmas=sigmas
        ).sum()

        Z = np.array([[filter(x, y) for x in X] for y in Y])
        sns.heatmap(Z, ax=ax, cbar=False)
        ax.set_xticks([])
        ax.set_yticks([])


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def compute_and_plot_heatmap(
    values,
    figax=None,
    log_scale=False,
    plot_f=False,
    random=True,
    minmax=(0, 1),
    smoothness=7,
    resolution=100,
    eps=1e-4,
    cbar=True,
    plot=True,
):

    x_values, y_values, z_values = filter_nans(values)

    if random:
        idxs = np.arange(len(x_values))
        np.random.shuffle(idxs)
        idxs = idxs[: len(idxs) // 10]
        values = x_values, y_values, z_values = (
            x_values[idxs],
            y_values[idxs],
            z_values[idxs],
        )

    X = np.linspace(x_values.min(), x_values.max(), resolution)
    Y = np.linspace(y_values.min(), y_values.max(), resolution)

    """
    # if not log_scale:
    #       # 500 x 500 takes 10s
    # else:
    #     X = np.linspace(x_values.min(), x_values.max(), resolution)
    #     # X = np.geomspace(np.maximum(x_values.min(), eps), x_values.max(), resolution)
    #     # Y = np.geomspace(np.maximum(y_values.min(), eps), y_values.max(), resolution)
    #     Y = np.linspace(y_values.min(), y_values.max(), resolution)
    #     # print(Y)
    #     # print(Y)
    """

    Xm, Ym = np.meshgrid(X, Y)

    # ratio = y_values / x_values
    # ratio = movingaverage(ratio, len(ratio) // 3)
    # sigmas = np.array([np.ones_like(ratio), ratio * 3]) * smoothness

    ratios = [(y_values / x_values).mean(), y_values.mean() / x_values.mean()]
    # print(ratios)
    ratio = ratios[1]
    sigmas = np.array([1, ratio])

    # ratio = x_values / y_values
    # sigmas = np.stack([ratio, np.ones_like(ratio)])

    try:
        sigmas[0] *= smoothness[0]
        sigmas[1] *= smoothness[1]
    except (TypeError, IndexError) as e:
        sigmas *= smoothness

    # vect_avg = np.vectorize(
    #     lambda x, y,: weighted_average(x, y, sigmas, values), signature=("(),()->()")
    # )
    vect_avg = np.vectorize(
        lambda x, y: weighted_average(x, y, sigmas, values), signature=("(),()->()")
    )

    Z = vect_avg(Xm, Ym)

    if plot_f:
        plot_filters(smoothness, values)

    if plot:

        if (figax) is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        else:
            fig, ax = figax

        pc = ax.pcolor(
            X,
            Y,
            Z,
            cmap="viridis",
            vmin=minmax[0] if minmax is not None else None,
            vmax=minmax[1] if minmax is not None else None,
            rasterized=True,
        )
        if log_scale:

            """"""
            # ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_ylim(y_values.min(), y_values.max())
        ax.set_xlim(x_values.min(), x_values.max())

        if cbar:
            cbar = fig.colorbar(pc, ax=ax)

    else:
        fig = ax = cbar = None

    return (X, Y), (Xm, Ym), Z, sigmas, (fig, ax), cbar


def compute_and_plot_colormesh(
    values, figax=None, method="nearest", log_scale=False, resolution=300, cbar=True
):

    x_values, y_values, z_values = values

    eps = 1e-4

    if not log_scale:
        X = np.linspace(x_values.min(), x_values.max(), resolution)
        Y = np.linspace(
            y_values.min(), y_values.max(), resolution
        )  # 500 x 500 takes 10s
    else:
        X = np.geomspace(np.maximum(x_values.min(), eps), x_values.max(), resolution)
        Y = np.geomspace(np.maximum(y_values.min(), eps), y_values.max(), resolution)
        # print(Y)

        # X = np.linspace(x_values.min(), x_values.max(), resolution)
        # Y = np.linspace(
        #    y_values.min(), y_values.max(), resolution
        # )  # 500 x 500 takes 10s

    X_mesh, Y_mesh = np.meshgrid(X, Y)

    Z = gd((x_values, y_values), z_values, (X_mesh, Y_mesh), method=method)
    if (figax) is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig, ax = figax

    fig, ax = figax
    # pcm = ax.pcolormesh(X_mesh, Y_mesh, Z, cmap="viridis")
    pcm = ax.pcolor(X, Y, Z, cmap="viridis", rasterized=True)
    # ax.set_ylim(y_values.min(), y_values.max())
    # ax.set_xlim(x_values.min(), x_values.max())
    if cbar:
        cbar = fig.colorbar(pcm, ax=ax)

    if log_scale:

        """"""
        # ax.set_xscale("log")
        ax.set_yscale("log")

    return X_mesh, Y_mesh, Z, (fig, ax), cbar
