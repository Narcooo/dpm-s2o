import os
import shutil
import torch
from torch.utils.data import Dataset
import random
from torchvision.transforms import Compose, ToTensor, Lambda

import numpy as np
import pandas as pd

transforms = Compose([
    # Resize(image_size),
    # CenterCrop(image_size),
    ToTensor(),# turn into Numpy array of shape HWC, divide by 255
    Lambda(lambda t: (t * 2) - 1),
])


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class RSD46feature(Dataset):
    def __init__(self, root_path, img_prefix='train', label_prefix=None, img_suffix='.jpg', transforms=None):
        super(RSD46feature, self).__init__()
        root = os.path.join(root_path)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        self.empty_token = 46
        self.p_uncond = 0.3
        self.images = _list_image_files_recursively(os.path.join(root))

        # self.labellist = sorted(os.listdir(os.path.join(root, label_prefix)))
        self.transforms = transforms




    def __getitem__(self, index):
        # image = Image.open(self.images[index]).convert("RGB")
        image = np.load(self.images[index])
        image = image.squeeze(0).transpose(1, 2, 0)
        label_str = self.images[index].split('/')[-1].split('_')[-1].split('.')[0]
        label = int(label_str)

        if self.transforms is not None:
            img_np = np.array(image)
            img_aug = self.transforms(image=img_np)
            image = img_aug['image']

        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = image / 0.18215 * 0.23908
        label = torch.tensor(label).long()
        if random.random() < self.p_uncond:
            label = torch.tensor(self.empty_token).long()

        return image, label

    def __len__(self):
        return len(self.images)

