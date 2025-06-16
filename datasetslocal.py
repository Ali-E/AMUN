

import os
from typing import *

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset


### from github
import pickle
from torch import FloatTensor, div
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
import numpy as np

### from salun
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
from tqdm import tqdm
import glob
from shutil import move
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
DATASET_LOC = './data'

os.environ[IMAGENET_LOC_ENV] = DATASET_LOC
# list of all datasets
DATASETS = ["imagenet", "cifar10", "mnist"]


def get_dataset(dataset: str, split: str) -> Dataset:
	"""Return the dataset as a PyTorch Dataset object"""
	if dataset == "imagenet":
		return _imagenet(split)
	elif dataset == "cifar10":
		return _cifar10(split)
	elif dataset == "mnist":
		return _mnist(split)


def get_num_classes(dataset: str):
	"""Return the number of classes in the dataset. """
	if dataset == "imagenet":
		return 1000
	elif dataset == "cifar10":
		return 10
	elif dataset == "mnist":
		return 10

def unnormalize(x: torch.tensor, dataset: str):
	if dataset == "imagenet":
		mean = torch.tensor(_IMAGENET_MEAN).to(device)
		stddev = torch.tensor(_IMAGENET_STDDEV).to(device)
	elif dataset == "cifar":
		mean = torch.tensor(_CIFAR10_MEAN).to(device)
		stddev = torch.tensor(_CIFAR10_STDDEV).to(device)
	elif dataset == "mnist":
		mean = torch.tensor(_MNIST_MEAN).to(device)
		stddev = torch.tensor(_MNIST_STDDEV).to(device)
	return x * stddev.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
	# return x * stddev + mean

def get_normalize_layer(dataset: str) -> torch.nn.Module:
	"""Return the dataset's normalization layer"""
	if dataset == "imagenet":
		return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
	elif dataset == "cifar10":
		return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
	elif dataset == "mnist":
		return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.5, ]
_MNIST_STDDEV = [0.5, ]

def _mnist(split: str) -> Dataset:
	if split == "train":
		return datasets.MNIST(DATASET_LOC, train=True, download=True, transform=transforms.Compose([
		transforms.RandomCrop(28),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
	]))
	elif split == "test":
		return datasets.MNIST(DATASET_LOC, train=False, transform=transforms.ToTensor())


def _cifar10(split: str) -> Dataset:
	if split == "train":
		return datasets.CIFAR10(DATASET_LOC, train=True, download=True, transform=transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()
		]))
	elif split == "test":
		return datasets.CIFAR10(DATASET_LOC, train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
	if not IMAGENET_LOC_ENV in os.environ:
		raise RuntimeError("environment variable for ImageNet directory not set")

	dir = os.environ[IMAGENET_LOC_ENV]
	if split == "train":
		subdir = os.path.join(dir, "train")
		transform = transforms.Compose([
			transforms.RandomSizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()
		])
	elif split == "test":
		subdir = os.path.join(dir, "val")
		transform = transforms.Compose([
			transforms.Scale(256),
			transforms.CenterCrop(224),
			transforms.ToTensor()
		])
	return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
	"""Standardize the channels of a batch of images by subtracting the dataset mean
	  and dividing by the dataset standard deviation.

	  In order to certify radii in original coordinates rather than standardized coordinates, we
	  add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
	  layer of the classifier rather than as a part of preprocessing as is typical.
	  """

	def __init__(self, means: List[float], sds: List[float]):
		"""
		:param means: the channel means
		:param sds: the channel standard deviations
		"""
		super(NormalizeLayer, self).__init__()
		self.means = torch.tensor(means).to(device)
		self.sds = torch.tensor(sds).to(device)

	def forward(self, input: torch.tensor):
		(batch_size, num_channels, height, width) = input.shape
		means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
		sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
		#print(input)
		return (input - means) / sds



class ImageNetDataset(Dataset):
    """Dataset class for ImageNet"""
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(ImageNetDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        data = div(data.type(FloatTensor), 255)
        if self.normalize:
            data = self.normalize(data)

        return data, self.labels[idx]

def load_train_data(img_size, magnitude, batch_size):
    with open('train_dataset.pkl', 'rb') as f:
        train_data, train_labels = pickle.load(f)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandAugment(num_ops=2,magnitude=magnitude),
    ])
    train_dataset = ImageNetDataset(train_data, train_labels.type(torch.LongTensor), transform,
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    f.close()
    return train_loader

def load_val_data(img_size, batch_size):
    with open('val_dataset.pkl', 'rb') as f:
        val_data, val_labels = pickle.load(f)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
    ])
    val_dataset = ImageNetDataset(val_data, val_labels.type(torch.LongTensor), transform,
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    f.close()
    return val_loader






def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = -dataset._labels[indexes] - 1


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = False,
):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)



class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start:end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path = os.path.join(args.data_dir, "val/")
        self.test_path = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/tiny",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        if class_to_replace is not None and indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        if class_to_replace is not None:
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
            if num_indexes_to_replace is None or num_indexes_to_replace == 500:
                test_set.targets = np.array(test_set.targets)
                test_set.imgs = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(
                dataset=train_set,
                indexes=indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

        loader_args = {"num_workers": 0, "pin_memory": False}

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        val_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader
