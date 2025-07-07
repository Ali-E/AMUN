import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import *
import pandas as pd
import copy
from torch.utils.data import Dataset
import numpy as np


def get_data(model_count=128, seed=0, dataset_size=40000, pkeep=0.5):
    np.random.seed(seed)
    keep = np.random.uniform(0,1,size=(model_count, dataset_size))
    order = keep.argsort(0)
    keep = order < int(pkeep * model_count)
    # keep = np.array(keep, dtype=bool)
    return keep


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def discretize(x):
    return torch.round(x * 255) / 255


def FGSM_perturb(x, y, model=None, bound=None, criterion=None):
    device = model.parameters().__next__().device
    model.zero_grad()
    x_adv = x.detach().clone().requires_grad_(True).to(device)
    pred = model(x_adv)
    loss = criterion(pred, y)
    loss.backward()
    grad_sign = x_adv.grad.data.detach().sign()
    x_adv = x_adv + grad_sign * bound
    x_adv = discretize(torch.clamp(x_adv, 0.0, 1.0))
    return x_adv.detach()


def expand_model(model):
    last_fc_name = None
    last_fc_layer = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_fc_name = name
            last_fc_layer = module

    if last_fc_name is None:
        raise ValueError("No Linear layer found in the model.")

    num_classes = last_fc_layer.out_features

    bias = last_fc_layer.bias is not None

    new_last_fc_layer = nn.Linear(
        in_features=last_fc_layer.in_features,
        out_features=num_classes + 1,
        bias=bias,
        device=last_fc_layer.weight.device,
        dtype=last_fc_layer.weight.dtype,
    )

    with torch.no_grad():
        new_last_fc_layer.weight[:-1] = last_fc_layer.weight
        if bias:
            new_last_fc_layer.bias[:-1] = last_fc_layer.bias

    parts = last_fc_name.split(".")
    current_module = model
    for part in parts[:-1]:
        current_module = getattr(current_module, part)
    setattr(current_module, parts[-1], new_last_fc_layer)


def imshow(img, path=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    plt.savefig(path)


class simpleDataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

        self.data = self.data.detach().cpu().numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = image.transpose(1, 2, 0)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomImageDataset(Dataset):
    def __init__(self, labels_file, imgs_path, unlearn_indices=None, transform=None, target_transform=None, num_classes=10, ablation_RL=False, ablation_RS=False, ablation_forgetset_RL=False, ablation_forgetset_AdvL=False, image_orig=None):#, device='cuda'):
        self.imgs_path = imgs_path
        self.num_classes = num_classes
        self.ablation_RL = ablation_RL
        self.ablation_forgetset_RL = ablation_forgetset_RL
        self.images_delta_df = pd.read_csv(labels_file)
        self.img_labels = self.images_delta_df['adv_pred'].values[unlearn_indices]
        self.img_deltas = self.images_delta_df['delta_norm'].values[unlearn_indices]
        self.transform = transform # feature transformation
        self.target_transform = target_transform # label transformation
        if not ablation_RS and not ablation_forgetset_RL and not ablation_forgetset_AdvL:
            self.adv_images = torch.load(self.imgs_path, map_location=torch.device('cpu'))
            self.adv_images = self.adv_images[unlearn_indices]
        else:
            self.adv_images = image_orig

        if ablation_RL or ablation_forgetset_RL:
            self.true_labels = self.images_delta_df['label'].values[unlearn_indices]

        if ablation_RS:
            ## choose adv images to be img_delta radius away from adv_images. In this case
            ## the adv_images images are the same as the original images not adv images.
            rand_vectors = torch.randn_like(self.adv_images)
            rand_vectors_norm = torch.norm(rand_vectors, p=2, dim=(1,2,3), keepdim=False)
            unit_vectors = rand_vectors / rand_vectors_norm.view(-1, 1, 1, 1)
            scaled_unit_vectors = unit_vectors * torch.tensor(self.img_deltas).view(-1, 1, 1, 1)
            check_norms = torch.norm(scaled_unit_vectors, p=2, dim=(1,2,3), keepdim=False)
            self.adv_images = self.adv_images + scaled_unit_vectors
            self.adv_images = torch.clamp(self.adv_images, min=0, max=1)

        self.adv_images = self.adv_images.detach().numpy()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.adv_images[idx]
        image = image.transpose(1, 2, 0)
        label = self.img_labels[idx]
        if self.ablation_RL or self.ablation_forgetset_RL:
            ## choose a random label other than the true label and adv label:
            label = np.random.choice([i for i in range(self.num_classes) if i != self.true_labels[idx] and i != self.img_labels[idx]]) 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class RLDataset(Dataset):
    def __init__(self, forgetset, new_classes=None, num_classes=10, noise_level=0.01, add_noise=False):
        self.image_set = forgetset
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.num_classes = num_classes
        self.new_classes = new_classes

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        image = self.image_set[idx][0]
        if self.new_classes is not None:
            label = self.new_classes[idx]
        else:
            true_label = self.image_set[idx][1]
            label = np.random.choice([i for i in range(self.num_classes) if i != true_label]) # random label

        if self.add_noise:
            noise = torch.randn_like(image) * self.noise_level
            adv_images = self.adv_images + noise
            adv_images = torch.clamp(adv_images, min=0, max=1)

        return image, label


class basicDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data.shape[-1] == 2:
            image_in = self.data[idx]['image']
            image = copy.deepcopy(np.asarray(image_in))
            if len(image.shape) == 2:
                image = copy.deepcopy(np.stack((image, image, image), axis=2))
        else:
            print('shape is 1')
            image_in = self.data[idx][0]

        if self.data.shape[-1] == 2:
            label = self.data[idx]['label']
        else:
            label = self.data[idx][1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label