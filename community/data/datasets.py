import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import EMNIST
import numpy as np
import numpy.random as rd
from typing import Any, Callable, Dict, List, Optional


class Custom_EMNIST(datasets.EMNIST) : 
    def __init__(self, root: str, train: bool = True,data_type: str = 'digits',
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 download: bool = False, truncate : list = None) -> None:
        
        self.truncate = truncate        
        super().__init__(root, train=train,
                               transform=transform,
                               target_transform=target_transform,
                               download=download, 
                               split=data_type) 

    def _load_data(self):
        data, targets = super()._load_data()
        if self.split == 'letters' : 
            targets -= 1
        if self.truncate is not None: 
            try : 
                truncate_mask = np.array(targets)<self.truncate
                mul = 10//self.truncate   
                truncate_values = np.arange(self.truncate)
            except ValueError : 
                truncate_mask = np.stack([np.array(targets) == t for t in self.truncate]).sum(0).clip(0, 1) == 1
                mul = 10//len(self.truncate)
                truncate_values = self.truncate

            data,  targets = torch.cat([data[truncate_mask]]*mul, 0), torch.cat([targets[truncate_mask]]*mul, 0)

            for i, t in enumerate(truncate_values) : 
                targets[targets == t] = i
            
            self.truncate_values = truncate_values
            
        return data, targets


class DoubleMNIST(Dataset) :
    """
    Double Digits MNIST dataset
    Args : 
        train : use training set
        asym : solve parity task asymetry by removing digit_1==digit_2 cases
    
    """
    
    def __init__(self, root, train=True, fix_asym=True, permute=False, seed=None):
        super().__init__()

        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        dataset = datasets.MNIST(root, train=train, download=True,
                   transform=self.transform)
        self.mnist_dataset = dataset

        self.fix_asym=fix_asym
        self.secondary_index = torch.randperm(len(self.mnist_dataset))
        if self.fix_asym : 
            self.new_idxs = self.get_forbidden_indexs()
        else : 
            self.new_idxs = [i for i in range(len(dataset))]
        
        self.permute = permute
        if self.permute : 
            if seed is not None : 
                torch.manual_seed(seed)
            self.permutation = torch.randperm(10)
            print(self.permutation)

    def valid_idx(self, idx) : 
        idx1, idx2 = idx, self.secondary_index[idx]
        _, target_1 = self.mnist_dataset[idx1]
        _, target_2 = self.mnist_dataset[idx2]

        return not (target_1 == target_2 or target_1 == (target_2-1)%10)

    def permute_labels(self, seed=None) : 
        if seed is not None : 
            torch.manual_seed(seed)
        self.permutation = torch.randperm(10)

    def get_forbidden_indexs(self) : 
        new_idxs = []
        for idx in range(len(self.mnist_dataset)) : 
            if self.valid_idx(idx) :
                new_idxs.append(idx)
        return new_idxs

    def __getitem__(self, index):
        index_1= self.new_idxs[index]
        index_2 = self.secondary_index[index_1]

        digit_1, target_1 = self.mnist_dataset[index_1]
        digit_2, target_2 = self.mnist_dataset[index_2]

        if self.permute : 
            target_1, target_2 = self.permutation[target_1],  self.permutation[target_2]

        digits = torch.cat([digit_1, digit_2], axis = 0)
        targets = [target_1, target_2]

        return digits, torch.tensor(targets)

    def __len__(self) : 
        return len(self.mnist_dataset)

    
class MultiDataset(Dataset) : 
    def __init__(self, datasets, shuffle=False, fix_asym=False) :
        
        self.datasets = datasets
        self.small_dataset_idx = np.argmin([len(d) for d in self.datasets])
        self.small_dataset = datasets[self.small_dataset_idx]
        self.shuffle = shuffle

        self.fix_asym=fix_asym
        self.secondary_index = torch.randperm(len(self.small_dataset))
        
        if self.fix_asym : 
            self.new_idxs = self.get_forbidden_indexs()
        else : 
            self.new_idxs = [i for i in range(len(self.small_dataset))]
        
    def valid_idx(self, idx) : 
            idx1, idx2 = idx, self.secondary_index[idx]
            _, target_1 = self.mnist_dataset[idx1]
            _, target_2 = self.mnist_dataset[idx2]
    def get_forbidden_indexs(self) : 
            new_idxs = []
            for idx in range(len(self.mnist_dataset)) : 
                if self.valid_idx(idx) :
                    new_idxs.append(idx)
            return new_idxs

    def __len__(self):
        return  len(self.small_dataset)

    def __getitem__(self, idx):
        #get images and labels here 
        #returned images must be tensor
        #labels should be int 
        rand_idx = lambda d : rd.randint(len(d))
        if self.shuffle : 
            idxs = [(rand_idx(d) if n!=self.small_dataset_idx else idx) for n, d in enumerate(self.datasets)]
        else : 
            #if self.fix_asym : 
            idxs = [idx]*len(self.datasets)

        samples = [d[idx] for (d, idx) in zip(self.datasets, idxs)]
        datas, labels = [d[0] for d in samples], [d[1] for d in samples]
        try : 
            return torch.stack(datas), torch.tensor(labels)
        except : 
            return datas, labels


def get_datasets(root, batch_size=256, use_cuda=True, fix_asym=False, permute=False, seed=None) :
        
    train_kwargs = {'batch_size': batch_size, 'shuffle' : True}
    test_kwargs = {'batch_size': batch_size, 'shuffle' : True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                    'pin_memory': True}
                    
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    kwargs = train_kwargs, test_kwargs

    single_datasets = [datasets.MNIST(root, train=t, download=True,
                    transform=transform) for t in [True, False]]
    double_datasets = [DoubleMNIST(root, train=t, fix_asym=False)
                        for t in [True, False]]

    single_datasets_fashion = [datasets.FashionMNIST(root, train=t,
                                 transform=transform, download=True)
                                  for t in [True, False]]

    transform=transforms.Compose([
        lambda img : transforms.functional.rotate(img, -90),
        lambda img : transforms.functional.hflip(img),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    truncate = truncate=np.arange(10, 21)
    truncate = truncate[truncate != 18]
    single_datasets_letters = [Custom_EMNIST(root, train=t, data_type='byclass',
                                 truncate=truncate, transform=transform) for t in [True, False]]

    multi_datasets = [MultiDataset([s1, s2]) for (s1, s2) in zip(single_datasets, single_datasets_letters)]

    single_loaders = [torch.utils.data.DataLoader(d, **k) for d, k in zip(single_datasets, kwargs)]
    double_loaders = [torch.utils.data.DataLoader(d, **k) for d, k in zip(double_datasets, kwargs)]
    multi_loaders = [torch.utils.data.DataLoader(d, **k) for d, k in zip(multi_datasets, kwargs)]

    return multi_loaders, double_loaders, single_loaders


