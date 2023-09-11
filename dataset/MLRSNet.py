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

class MLRSNet(Dataset):
    def __init__(self, root_path, img_prefix='Images', label_prefix='Labels', img_suffix='.jpg', transforms=None):
        super(MLRSNet, self).__init__()
        root = os.path.join(root_path)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        # image_dir = os.path.join(root, img_prefix)
        # label_dir = os.path.join(root, label_prefix)
        # self.images = sorted(glob.glob(os.path.join(image_dir, '*' + img_suffix)))
        # self.images = _list_image_files_recursively(os.path.join(root, img_prefix))
        self.labellist = sorted(os.listdir(os.path.join(root, label_prefix)))
        # class_names = [os.path.basename(path).split("_")[0] for path in self.images]
        # sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        # classes = [sorted_classes[x] for x in class_names]
        self.images = []
        self.labels = []
        for file in self.labellist:
            df = pd.read_csv(os.path.join(root, label_prefix, file), index_col=0)
            iname = df.index.tolist()
            iname = [os.path.join(root, img_prefix, x.split('.')[0][:-6], x) for x in iname]
            self.images.extend(iname)
            ln = df.values.tolist()
            self.labels.extend(ln)
            # break
        # self.labels = classes
        self.transforms = transforms
        # img_listdir = os.listdir(image_dir)
        # nn = 0
        # # print("Loading Dataset...\n")
        # for class_name in img_listdir:
        #     class_dir = os.path.join(image_dir, class_name)
        #     df = pd.read_csv(os.path.join(label_dir, class_name + '.csv'))
        #     class_listdir = os.listdir(class_dir)
        #     nn += 1
        #     for image_name in (class_listdir):
        #         image_path = os.path.join(class_dir, image_name)
        #         # image = Image.open(image_path)
        #         self.images.append(image_path)
        #
        #         df_cl_name = df[df['image'] == image_name]
        #         label = df_cl_name.values[:, 1:].astype(np.float)
        #         label = np.squeeze(label)
        #         self.labels.append(label)
        #
        #     # if nn >= 1:
        #     #     break
        # a=1


    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        # img = cv2.imread(self.images[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        if self.transforms is not None:
            img_np = np.array(image)
            img_aug = self.transforms(image=img_np)
            image = img_aug['image']
            #
            # img_np_w = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("/data/MLRSNet/img/image.jpg", img_np_w)
            # image_w = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("/data/MLRSNet/img/image_aug.jpg", image_w)

        image = transforms(image)
        label = torch.tensor(label).float()
            # dict = {
            #     'img': image,
            #     'label': label
            # }
            # a=1

        return image, label
    def __len__(self):
        return len(self.images)

