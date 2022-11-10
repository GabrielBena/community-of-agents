from random import seed as seed_r
from numpy.random import seed as seed_np
from os import environ
from torch.random import manual_seed as seed_tcpu
from torch.cuda import manual_seed_all as seed_tgpu
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm

# ------ Others ------


def rescue_code(function):
    import inspect

    get_ipython().set_next_input("".join(inspect.getsourcelines(function)[0]))


def set_seeds(seed=42):
    seed_r(seed)
    seed_np(seed)
    environ["PYTHONHASHSEED"] = str(seed)
    seed_tcpu(seed)
    seed_tgpu(seed)


def check_grad(model, task_id="0"):
    for n, p in model.named_parameters():
        if "k_params" in n or "all_scores" in n:
            if task_id in n:
                return check_ind_grad(n, p)
        else:
            check_ind_grad(n, p)


def check_ind_grad(n, p):
    if p.grad is not None:
        if (p.grad == 0).all():
            """"""
            print(f"{n}, Zero Grad")
        # else : print(f'{n} : {p.grad}')
    elif p.requires_grad:
        """"""
        print(f"{n}, None Grad")


def is_notebook():
    try:
        get_ipython()
        notebook = True
    except NameError:
        notebook = False
    return notebook


def tqdm_module():

    if is_notebook():
        return tqdm_n
    else:
        return tqdm
