from community.data.tasks import get_task_target
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import shutil, os


# ------ Plotting utils ------


def plot_running_data(data, ax=None, m=1, **plot_kwargs):
    try:
        x, metric = data
    except ValueError:
        x, metric = range(len(data)), data
    if ax is None:
        ax = plt.subplot(111)
    running = np.convolve(metric, np.ones((m)) / (m), mode="valid")
    running_x = np.convolve(x, np.ones((m)) / (m), mode="valid")
    ax.plot(running_x, running, **plot_kwargs)
    plt.legend()


def plot_grid(
    imgs,
    labels=None,
    row_title=None,
    figsize=None,
    save_loc=None,
    colorbar=False,
    **imshow_kwargs,
):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]
        if labels is not None:
            labels = [labels]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_cols, squeeze=False, figsize=figsize
    )
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            im = ax.imshow(np.asarray(img[0]), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if labels is not None:
                ax.set_title(labels[row_idx][col_idx])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    if colorbar:
        fig.colorbar(im)

    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc)
    plt.show()


def create_gifs(data, target=None, name="gif", input_size=30, task=None, double=True):

    if task:
        target = get_task_target(target, task)

    try:
        shutil.rmtree("gifs/")
        os.mkdir("gifs")
    except FileNotFoundError:
        os.mkdir("gifs")
        "continue"

    img_list = lambda i: data[..., i, :, :].cpu().data.numpy() * 255

    def create_gif(img_list, l, w, name, double=True):

        images_list = [
            Image.fromarray(img.reshape(w, l)).resize((256, 256)) for img in img_list
        ]
        images_list = images_list  # + images_list[::-1] # loop back beginning

        images_list[0].save(
            f"{name}.gif", save_all=True, append_images=images_list[1:], loop=0
        )

    data_size = data.shape[2]
    if target is None:
        target = ["" for _ in range(data_size)]
    else:
        target = target.cpu().data.numpy()

    if double:
        [
            create_gif(
                img_list(i),
                input_size,
                2 * input_size,
                "gifs/" + f"{name}_{i}_{target[i]}",
            )
            for i in range(min(10, len(target)))
        ]
    else:
        [
            create_gif(
                img_list(i), input_size, input_size, "gifs/" + f"{name}_{i}_{target[i]}"
            )
            for i in range(min(10, len(target)))
        ]
