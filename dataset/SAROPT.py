import os
import shutil
import torch
from torch.utils.data import Dataset
import glob
import cv2
import tqdm
from torchvision.transforms import Compose, ToTensor, Lambda
from PIL import Image
import numpy as np
import pandas as pd
from dataset.transform import C_trans as opttrans
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
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class SAROPT(Dataset):
    def __init__(self, root_path, img_prefix='Images', label_prefix='Labels', img_suffix='.jpg', transforms=None):
        super(SAROPT, self).__init__()
        root = os.path.join(root_path)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        # image_dir = os.path.join(root, img_prefix)
        # label_dir = os.path.join(root, label_prefix)
        # self.images = sorted(glob.glob(os.path.join(image_dir, '*' + img_suffix)))
        self.images = _list_image_files_recursively(os.path.join(root, img_prefix))
        self.labels = _list_image_files_recursively(os.path.join(root, label_prefix))
        # class_names = [os.path.basename(path).split("_")[0] for path in self.images]
        # sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        # classes = [sorted_classes[x] for x in class_names]
            # break
        # self.labels = classes
        self.transforms = transforms



    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        # image = Image.open(self.images[index])
        # image = np.array(image)

        label = Image.open(self.labels[index]).convert('RGB')
        # label = np.array(label)
        if self.transforms is not None:
            img_np = np.array(image)
            label_np = np.array(label)
            img_aug = self.transforms(image=img_np, mask=label_np)
            image = img_aug['image']
            #
            # image = opttrans(image=image)['image']
            label = img_aug['mask']
            a=1
            # img_np_w = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("/data/MLRSNet/img/image.jpg", img_np_w)
            # image_w = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("/data/MLRSNet/img/image_aug.jpg", image_w)

        image = transforms(image)
        label = transforms(label)

            # dict = {
            #     'img': image,
            #     'label': label
            # }
            # a=1

        return image, label, self.labels[index].split('/')[-1].split('.')[0]
    def __len__(self):
        return len(self.images)

